"""
test_guardrails.py
Unit tests and security verification for RiskGuardrails and SQLEngine.
Ensures 100% of malicious DDL/DML operations and SQL Injections are blocked.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.text_to_sql.risk_guardrails import RiskGuardrails
from src.text_to_sql.execution_engine import SQLEngine, SecurityViolationError


@pytest.fixture
def guardrails():
    return RiskGuardrails()


@pytest.fixture
def sql_engine():
    db_path = "data/database/finance.db"
    return SQLEngine(db_path=db_path)


# =====================================================================
# TEST CASE 1: Câu lệnh SELECT hợp lệ 1 (Truy vấn đơn giản với LIMIT)
# =====================================================================
def test_valid_select_query_simple(guardrails, sql_engine):
    query = "SELECT * FROM table_0 LIMIT 5;"
    is_safe, reason = guardrails.validate_query(query)
    assert is_safe is True
    assert "hợp lệ" in reason.lower()

    # Thực thi qua SQLEngine
    df = sql_engine.execute(query)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 5


# =====================================================================
# TEST CASE 2: Câu lệnh SELECT hợp lệ 2 (Lọc có WHERE, ORDER BY, metadata)
# =====================================================================
def test_valid_select_query_with_filters(guardrails, sql_engine):
    query = """
    SELECT row_id, item_name 
    FROM table_0 
    WHERE row_id >= 0 
    ORDER BY row_id ASC 
    LIMIT 3;
    """
    is_safe, reason = guardrails.validate_query(query)
    assert is_safe is True

    # Thực thi qua SQLEngine
    df = sql_engine.execute(query)
    assert isinstance(df, pd.DataFrame)
    assert "row_id" in df.columns
    assert len(df) <= 3


# =====================================================================
# TEST CASE 3: Lệnh phá hoại DROP TABLE (Phải bị chặn tuyệt đối)
# =====================================================================
def test_block_drop_table(guardrails, sql_engine):
    query = "DROP TABLE table_0;"
    is_safe, reason = guardrails.validate_query(query)
    assert is_safe is False
    assert "DROP" in reason or "không được phép" in reason

    # Đảm bảo SQLEngine ném ra SecurityViolationError
    with pytest.raises(SecurityViolationError) as exc_info:
        sql_engine.execute(query)
    assert "vi phạm an ninh" in str(exc_info.value)


# =====================================================================
# TEST CASE 4: Lệnh sửa đổi dữ liệu UPDATE (Phải bị chặn tuyệt đối)
# =====================================================================
def test_block_update_statement(guardrails, sql_engine):
    query = "UPDATE table_0 SET col_1 = 'HACKED' WHERE row_id = 0;"
    is_safe, reason = guardrails.validate_query(query)
    assert is_safe is False
    assert "UPDATE" in reason or "không được phép" in reason

    with pytest.raises(SecurityViolationError) as exc_info:
        sql_engine.execute(query)
    assert "vi phạm an ninh" in str(exc_info.value)


# =====================================================================
# TEST CASE 5: Tấn công SQL Injection nối chuỗi '; DROP TABLE' (Stacked Query)
# =====================================================================
def test_block_sql_injection_stacked_drop(guardrails, sql_engine):
    query = "SELECT * FROM table_0 WHERE row_id = 1; DROP TABLE table_1;"
    is_safe, reason = guardrails.validate_query(query)
    assert is_safe is False
    assert ("nhiều câu lệnh" in reason.lower() or "stacked" in reason.lower() or "drop" in reason.lower())

    with pytest.raises(SecurityViolationError) as exc_info:
        sql_engine.execute(query)
    assert "vi phạm an ninh" in str(exc_info.value)


# =====================================================================
# TEST CASE BỔ SUNG: Chặn các thao tác nguy hại khác (DELETE, INSERT, ALTER, TRUNCATE)
# =====================================================================
@pytest.mark.parametrize("malicious_query, blocked_op", [
    ("DELETE FROM table_0 WHERE row_id = 1;", "DELETE"),
    ("INSERT INTO table_0 (item_name) VALUES ('MALICIOUS');", "INSERT"),
    ("ALTER TABLE table_0 ADD COLUMN malicious TEXT;", "ALTER"),
    ("TRUNCATE TABLE table_0;", "TRUNCATE"),
    ("GRANT ALL PRIVILEGES ON table_0 TO public;", "GRANT"),
    ("REVOKE ALL PRIVILEGES ON table_0 FROM public;", "REVOKE"),
    ("SELECT * FROM table_0 WHERE item_name = (SELECT 1 FROM sqlite_master); DELETE FROM table_0;", "DELETE"),
])
def test_block_other_malicious_operations(guardrails, sql_engine, malicious_query, blocked_op):
    is_safe, reason = guardrails.validate_query(malicious_query)
    assert is_safe is False

    with pytest.raises(SecurityViolationError):
        sql_engine.execute(malicious_query)
