"""
test_sql_generator.py
Unit tests for NL2SQLGenerator with Mock LLM responses.
Verifies end-to-end Text-to-SQL prompt creation, execution, and security blocking.
"""

from unittest.mock import MagicMock
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from src.text_to_sql.sql_generator import NL2SQLGenerator
from src.text_to_sql.execution_engine import SecurityViolationError


class MockLLM:
    """
    Mock LLM giả lập phản hồi của ChatOllama / BaseChatModel
    để kiểm thử chính xác các kịch bản sinh SQL khác nhau.
    """

    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke(self, messages):
        return AIMessage(content=self.response_text)


@pytest.fixture
def db_path():
    return "data/database/finance.db"


# ======================================================================
# TEST 1: Kiểm tra hàm create_sql_prompt() tự động trích xuất schema
# ======================================================================
def test_create_sql_prompt_contains_schema(db_path):
    mock_llm = MockLLM("SELECT 1;")
    generator = NL2SQLGenerator(db_path=db_path, llm=mock_llm, sample_tables=["table_0"])

    messages = generator.create_sql_prompt("Thống kê số lượng bảng tài chính?")
    assert len(messages) == 2

    system_msg = messages[0].content
    user_msg = messages[1].content

    # Kiểm tra prompt có chứa schema và các quy tắc tài chính
    assert "DATABASE SCHEMA" in system_msg
    assert "table_0" in system_msg
    assert "SELECT" in system_msg
    assert "SUM()" in system_msg or "GROUP BY" in system_msg
    assert "Thống kê số lượng bảng tài chính?" in user_msg


# ======================================================================
# TEST 2: Kiểm tra hàm clean_sql_output() loại bỏ markdown block
# ======================================================================
def test_clean_sql_output(db_path):
    mock_llm = MockLLM("SELECT 1;")
    generator = NL2SQLGenerator(db_path=db_path, llm=mock_llm)

    raw_markdown = "```sql\nSELECT * FROM table_0 LIMIT 5;\n```"
    cleaned = generator.clean_sql_output(raw_markdown)
    assert cleaned == "SELECT * FROM table_0 LIMIT 5;"

    raw_with_prefix = "SQL: SELECT item_name FROM table_0"
    cleaned_prefix = generator.clean_sql_output(raw_with_prefix)
    assert cleaned_prefix == "SELECT item_name FROM table_0"


# ======================================================================
# TEST 3: Luồng thành công - Sinh câu lệnh SELECT hợp lệ và thực thi
# ======================================================================
def test_generate_and_execute_valid_select(db_path):
    # Giả lập LLM sinh câu lệnh SELECT hợp lệ
    mock_sql = "SELECT row_id, item_name FROM table_0 LIMIT 3;"
    mock_llm = MockLLM(mock_sql)

    generator = NL2SQLGenerator(db_path=db_path, llm=mock_llm)
    result = generator.generate_and_execute("Lấy 3 mục đầu tiên trong table_0?")

    assert result["status"] == "success"
    assert result["is_safe"] is True
    assert result["sql_query"] == mock_sql
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) == 3
    assert "row_id" in result["data"].columns


# ======================================================================
# TEST 4: Luồng thành công với hàm tổng hợp GROUP BY / COUNT / SUM
# ======================================================================
def test_generate_and_execute_aggregation_query(db_path):
    mock_sql = """
    SELECT item_name, COUNT(*) as count_items 
    FROM table_0 
    GROUP BY item_name 
    LIMIT 5;
    """
    mock_llm = MockLLM(f"```sql\n{mock_sql}\n```")

    generator = NL2SQLGenerator(db_path=db_path, llm=mock_llm)
    result = generator.generate_and_execute("Thống kê số lượng mục trong table_0?")

    assert result["status"] == "success"
    assert result["is_safe"] is True
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) > 0
    assert "count_items" in result["data"].columns


# ======================================================================
# TEST 5: Luồng bảo mật - Chặn tuyệt đối nếu LLM sinh câu lệnh DROP TABLE
# ======================================================================
def test_block_malicious_drop_table_generated(db_path):
    # Giả lập tình huống LLM bị jailbreak hoặc sinh lệnh nguy hiểm
    mock_malicious_sql = "DROP TABLE table_0;"
    mock_llm = MockLLM(mock_malicious_sql)

    generator = NL2SQLGenerator(db_path=db_path, llm=mock_llm)
    result = generator.generate_and_execute("Xóa toàn bộ dữ liệu của table 0")

    assert result["status"] == "security_violation"
    assert result["is_safe"] is False
    assert result["data"] is None
    assert "DROP" in result["reason"] or "không được phép" in result["reason"]

    # Kiểm tra cờ raise_on_error=True ném ra SecurityViolationError
    with pytest.raises(SecurityViolationError) as exc_info:
        generator.generate_and_execute("Xóa toàn bộ dữ liệu", raise_on_error=True)
    assert "RỦI RO AN NINH" in str(exc_info.value)


# ======================================================================
# TEST 6: Luồng bảo mật - Chặn SQL Injection dạng Stacked Query '; DROP TABLE'
# ======================================================================
def test_block_sql_injection_stacked_query(db_path):
    mock_injection_sql = "SELECT * FROM table_0; DROP TABLE table_1;"
    mock_llm = MockLLM(mock_injection_sql)

    generator = NL2SQLGenerator(db_path=db_path, llm=mock_llm)
    result = generator.generate_and_execute("Truy vấn kèm injection")

    assert result["status"] == "security_violation"
    assert result["is_safe"] is False
    assert result["data"] is None
    assert "Stacked" in result["reason"] or "nhiều câu lệnh" in result["reason"] or "DROP" in result["reason"]


# ======================================================================
# TEST 7: Luồng bảo mật - Chặn lệnh UPDATE ngầm
# ======================================================================
def test_block_update_statement_generated(db_path):
    mock_update_sql = "UPDATE table_0 SET col_1 = 'compromised' WHERE row_id = 0;"
    mock_llm = MockLLM(mock_update_sql)

    generator = NL2SQLGenerator(db_path=db_path, llm=mock_llm)
    result = generator.generate_and_execute("Cập nhật lại số liệu bảng 0")

    assert result["status"] == "security_violation"
    assert result["is_safe"] is False
    assert result["data"] is None
    assert "UPDATE" in result["reason"] or "không được phép" in result["reason"]
