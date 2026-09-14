"""
test_evaluation.py
Unit tests for FinQAEvaluator and G-Eval LLM-as-a-Judge matching logic.
"""

import pytest
import pandas as pd
from langchain_core.messages import AIMessage
from src.pipelines.evaluation_pipeline import FinQAEvaluator
from src.agents.orchestrator import FinancialOrchestrator


class MockEvalLLM:
    """Mock LLM trả về SQL chuẩn giúp test benchmark chạy trong mili-giây."""
    def invoke(self, messages):
        return AIMessage(content="SELECT * FROM table_0 LIMIT 2;")


@pytest.fixture
def evaluator():
    mock_llm = MockEvalLLM()
    orchestrator = FinancialOrchestrator(
        db_path="data/database/finance.db",
        llm=mock_llm
    )
    return FinQAEvaluator(
        db_path="data/database/finance.db",
        orchestrator=orchestrator,
        sample_size=5
    )


def test_evaluator_dataset_loading(evaluator):
    """Kiểm tra việc load 5 câu hỏi từ finqa_metadata."""
    assert len(evaluator.test_dataset) == 5
    sample = evaluator.test_dataset[0]
    assert "question" in sample
    assert "exe_ans" in sample
    assert len(sample["exe_ans"]) > 0


def test_g_eval_numeric_matching(evaluator):
    """Kiểm tra logic so khớp số học chính xác và dung sai phần trăm."""
    # Khớp chính xác
    score, reason = evaluator.llm_as_a_judge(
        question="Interest expense?",
        generated_result=pd.DataFrame({"value": ["$3.8 million"]}),
        gold_answer="3.8"
    )
    assert score == 1
    assert "Khớp" in reason

    # Khớp dạng phần trăm (0.532 tương đương 53.2%)
    score_pct, _ = evaluator.llm_as_a_judge(
        question="Percentage?",
        generated_result=pd.DataFrame({"rate": ["53.2%"]}),
        gold_answer="0.532"
    )
    assert score_pct == 1

    # Sai lệch hoàn toàn
    score_fail, _ = evaluator.llm_as_a_judge(
        question="Revenue?",
        generated_result=pd.DataFrame({"value": ["$999,999"]}),
        gold_answer="12.5"
    )
    assert score_fail == 0


def test_evaluate_accuracy_strict_zero_trust(evaluator):
    """Kiểm tra evaluate_accuracy theo đúng chuẩn Zero-Trust."""
    # 1. None / SQL Error -> FAIL (0)
    s, r = evaluator.evaluate_accuracy(None, "100.0")
    assert s == 0
    assert "FAIL" in r

    # 2. Bảng rỗng -> FAIL (0)
    s, r = evaluator.evaluate_accuracy(pd.DataFrame(), "100.0")
    assert s == 0
    assert "FAIL" in r

    # 3. Lệch số > 1% -> FAIL (0)
    s, r = evaluator.evaluate_accuracy(pd.DataFrame({"res": [105.0]}), "100.0")
    assert s == 0
    assert "FAIL" in r

    # 4. Khớp số trong sai số 1% -> PASS (1)
    s, r = evaluator.evaluate_accuracy(pd.DataFrame({"res": [100.8]}), "100.0")
    assert s == 1
    assert "Khớp" in r


def test_benchmark_run_small_sample(evaluator):
    """Kiểm tra việc chạy benchmark trên tập mẫu 3 câu hỏi và tính KPI."""
    summary = evaluator.run_benchmark(sample_size=3, ignore_checkpoint=True)

    assert summary["total_evaluated"] == 3
    assert "accuracy_percentage" in summary
    assert "average_latency_seconds" in summary
    assert "time_saved_percentage" in summary
    assert summary["average_latency_seconds"] < 1.0
    assert summary["time_saved_percentage"] > 90.0
