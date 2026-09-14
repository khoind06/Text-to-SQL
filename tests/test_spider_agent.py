"""
tests/test_spider_agent.py
Unit tests cho Single-Agent SQL Architecture trên tập dữ liệu Spider.
"""

import pytest
from unittest.mock import MagicMock
from langchain_community.llms.fake import FakeListLLM

from src.agents.orchestrator import FinancialOrchestrator
from src.text_to_sql.sql_generator import NL2SQLGenerator
from src.text_to_sql.execution_engine import SQLEngine
from scripts.evaluate_spider import compare_results, categorize_error, normalize_val


def test_spider_schema_extraction():
    """Kiểm tra trích xuất Lược đồ đa bảng Spider chứa CREATE TABLE và FOREIGN KEY."""
    db_path = "data/spider/database/concert_singer/concert_singer.sqlite"
    schema = NL2SQLGenerator.extract_spider_schema(db_path)
    assert "CREATE TABLE" in schema
    assert "stadium" in schema
    assert "singer" in schema
    assert "concert" in schema
    assert "FOREIGN KEY" in schema


def test_single_agent_orchestrator_flow():
    """Kiểm tra luồng Single-Agent từ câu hỏi -> SQL -> Kết quả DB."""
    fake_llm = FakeListLLM(responses=["SELECT count(*) FROM singer;"])
    orchestrator = FinancialOrchestrator(llm=fake_llm)
    
    result = orchestrator.run(
        question="How many singers do we have?",
        db_id="concert_singer"
    )

    assert result.get("sql_query") == "SELECT count(*) FROM singer"
    assert result.get("sql_result") is not None
    assert len(result.get("sql_result")) == 1
    assert result.get("error") is None
    assert "sql_generator_ms" in result.get("latency_trace", {})
    assert "sql_executor_ms" in result.get("latency_trace", {})


def test_spider_execution_accuracy_comparison():
    """Kiểm tra hàm so sánh kết quả Execution Accuracy (EX)."""
    # 1. Trùng khớp không cần thứ tự
    gold_rows = [(1, "Alice"), (2, "Bob")]
    pred_rows = [(2, "Bob"), (1, "Alice")]
    assert compare_results(gold_rows, pred_rows, has_order_by=False) is True
    assert compare_results(gold_rows, pred_rows, has_order_by=True) is False

    # 2. Số lượng dòng lệch nhau
    assert compare_results(gold_rows, [(1, "Alice")], has_order_by=False) is False

    # 3. Ép kiểu số thực làm tròn
    gold_num = [(10.00001,)]
    pred_num = [(10.0,)]
    assert compare_results(gold_num, pred_num, has_order_by=False) is True


def test_error_categorization():
    """Kiểm tra phân loại lỗi đánh giá Spider."""
    assert categorize_error("no such table: singer", "SELECT * FROM singer") == "SCHEMA_ERROR"
    assert categorize_error("no such column: age", "SELECT age FROM singer") == "SCHEMA_ERROR"
    assert categorize_error("near 'WHERE': syntax error", "SELECT * WHERE") == "SYNTAX_ERROR"
    assert categorize_error("SecurityViolationError: Chặn DDL", "DROP TABLE singer") == "GUARDRAIL_BLOCKED"
    assert categorize_error("", "SELECT * FROM singer") == "LOGIC_ERROR"
