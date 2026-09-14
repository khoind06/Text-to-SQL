"""
test_sql_engine.py
Enterprise Unit tests for SQLEngine with Defense-in-Depth:
- Layer 1: AST RiskGuardrails validation
- Layer 2: SQLite C-Engine Read-Only URI (?mode=ro&uri=true)
"""

import pytest
import pandas as pd
from sqlalchemy import text
from src.text_to_sql.execution_engine import SQLEngine, SecurityViolationError
from src.text_to_sql.risk_guardrails import RiskGuardrails


@pytest.fixture
def db_path():
    return "data/database/finance.db"


@pytest.fixture
def sql_engine(db_path):
    return SQLEngine(db_path=db_path, read_only=True)


def test_safe_select_query(sql_engine):
    """Kiểm tra thực thi truy vấn SELECT hợp lệ trên CSDL Read-Only."""
    df = sql_engine.execute("SELECT row_id, item_name FROM table_0 LIMIT 3;")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "row_id" in df.columns


def test_defense_in_depth_sqlite_c_engine_blocks_write(sql_engine):
    """
    [CRITICAL DEFENSE-IN-DEPTH TEST]
    Giả định trường hợp xấu nhất: AST Guardrails bị bypass và cho phép lệnh DROP TABLE.
    C-Engine của SQLite dưới mode=ro PHẢI từ chối ghi và ném ra SecurityViolationError.
    """
    # Tạo mock guardrails bị bypass (cho phép mọi thứ)
    class BypassedGuardrails(RiskGuardrails):
        def validate_query(self, query: str):
            return True, "Bypassed for test"

    vulnerable_engine = SQLEngine(
        db_path=sql_engine.db_path,
        guardrails=BypassedGuardrails(),
        read_only=True
    )

    # Thử nghiệm thực thi lệnh DROP TABLE trên bảng table_0
    with pytest.raises(SecurityViolationError) as exc_info:
        vulnerable_engine.execute("DROP TABLE table_0;")

    # Xác nhận lỗi xuất phát từ tầng C-engine của SQLite Read-Only
    err_text = str(exc_info.value).lower()
    assert any(k in err_text for k in ["readonly database", "attempt to write", "defense-in-depth"])


def test_get_tables_and_schema(sql_engine):
    """Kiểm tra lấy danh sách bảng và cấu trúc schema qua inspect API."""
    tables = sql_engine.get_tables()
    assert len(tables) > 0
    assert "table_0" in tables

    columns = sql_engine.get_table_schema("table_0")
    assert len(columns) > 0
    col_names = [c["name"] for c in columns]
    assert "row_id" in col_names
    assert "item_name" in col_names
