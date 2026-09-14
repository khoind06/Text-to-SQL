"""
test_agents.py
Enterprise Unit tests for LangGraph Multi-Agent Financial System:
- SQL Generator Agent (FinancialPlannerAgent: pure SQL translation)
- SQLExecutorNode & AST RiskGuardrails (Execution & Security Isolation)
- Risk Analyst Agent (Quantitative Exposure & CRO Reporting)
- FinancialOrchestrator (StateGraph 4-Stage Decoupled Flow & Conditional Routing)
"""

import pytest
import pandas as pd
from langchain_core.messages import AIMessage

from src.agents.financial_planner import FinancialPlannerAgent
from src.agents.risk_analyst import RiskAnalystAgent
from src.agents.orchestrator import FinancialOrchestrator, AgentState


class MockAgentLLM:
    """Mock LLM trả về SQL, Python hoặc phân tích tùy theo input prompt."""
    def invoke(self, messages):
        user_content = messages[-1].content.lower()
        if any(k in user_content for k in ["mã python", "python", "dataframe `df`", "gán vào biến `result`"]):
            return AIMessage(content="result = 42.0")
        elif any(k in user_content for k in ["thẩm định", "cro", "rủi ro", "risk"]):
            return AIMessage(content="📊 Báo cáo Thẩm định Rủi ro: Mức độ Rủi ro Cao do phát hiện hợp đồng phái sinh thâm hụt. Score: 75/100.")
        elif any(k in user_content for k in ["drop", "xóa", "delete", "truncate"]):
            return AIMessage(content="DROP TABLE table_0;")
        elif any(k in user_content for k in ["table_0", "doanh thu"]):
            return AIMessage(content="SELECT item_name, amount FROM table_0 LIMIT 2;")
        else:
            return AIMessage(content="SELECT * FROM table_0 LIMIT 1;")


@pytest.fixture
def mock_llm():
    return MockAgentLLM()


@pytest.fixture
def db_path():
    return "data/database/finance.db"


# ======================================================================
# 1. Kiểm thử SQL Generator Agent (FinancialPlannerAgent - Pure NL2SQL)
# ======================================================================
def test_sql_generator_agent_pure_translation(db_path, mock_llm):
    """SQL Generator Agent chỉ đảm nhiệm dịch NL sang SQL, không can thiệp CSDL."""
    planner = FinancialPlannerAgent(db_path=db_path, llm=mock_llm)
    state = {"question": "Lấy thông tin doanh thu trong table_0"}

    result = planner.run_node(state)
    assert result["error"] is None
    assert "sql_query" in result
    assert "SELECT" in result["sql_query"]
    assert "table_0" in result["sql_query"]


def test_sql_generator_agent_empty_question(db_path, mock_llm):
    planner = FinancialPlannerAgent(db_path=db_path, llm=mock_llm)
    state = {"question": ""}

    result = planner.run_node(state)
    assert result["error"] is not None
    assert result["sql_query"] == ""


# ======================================================================
# 2. Kiểm thử Python Quant Engine & Zero-Trust Sandbox (Agent 2)
# ======================================================================
def test_python_quant_sandbox_ast_validation():
    from src.agents.python_quant_engine import PythonQuantSandbox

    # 1. Chặn import nguy hiểm
    is_safe, msg = PythonQuantSandbox.validate_ast("import os\nresult = os.system('ls')")
    assert not is_safe
    assert "Cấm import" in msg

    is_safe, msg = PythonQuantSandbox.validate_ast("from subprocess import Popen\nresult = 1")
    assert not is_safe

    # 2. Chặn hàm nguy hiểm
    is_safe, msg = PythonQuantSandbox.validate_ast("result = eval('2 + 2')")
    assert not is_safe
    assert "Cấm gọi hàm nguy hiểm" in msg

    is_safe, msg = PythonQuantSandbox.validate_ast("with open('secret.txt') as f: result = 1")
    assert not is_safe

    # 3. Cho phép mã toán học an toàn
    safe_code = "result = (df['col_a'].iloc[0] - df['col_b'].iloc[0]) / df['col_b'].iloc[0]"
    is_safe, msg = PythonQuantSandbox.validate_ast(safe_code)
    assert is_safe


def test_python_quant_sandbox_execution():
    from src.agents.python_quant_engine import PythonQuantSandbox
    df = pd.DataFrame({"rev_2023": [120.0], "rev_2022": [100.0]})

    code = "result = (df['rev_2023'].iloc[0] - df['rev_2022'].iloc[0]) / df['rev_2022'].iloc[0]"
    val, err = PythonQuantSandbox.execute(code, df)
    assert err is None
    assert pytest.approx(val, 0.001) == 0.2


def test_python_quant_agent_node(mock_llm):
    from src.agents.python_quant_engine import PythonQuantAgent
    agent = PythonQuantAgent(llm=mock_llm)
    df = pd.DataFrame({"col": [1, 2, 3]})
    state = {"question": "Tính tổng các phần tử", "sql_result": df}

    res = agent.run_node(state)
    assert res["error"] is None
    assert "result = 42.0" in res["python_code"]


# ======================================================================
# 3. Kiểm thử Risk Analyst Agent (Quantitative Analysis)
# ======================================================================
def test_risk_analyst_quantitative_node(mock_llm):
    analyst = RiskAnalystAgent(llm=mock_llm)
    dummy_df = pd.DataFrame({
        "item_name": ["Forward contract asset", "Unfavorable movement liability"],
        "amount": ["$10,000", "$(15,000)"]
    })
    state = {
        "question": "Đánh giá chi phí so với doanh thu",
        "sql_query": "SELECT * FROM table_0;",
        "sql_result": dummy_df
    }

    result = analyst.run_node(state)
    assert "analysis_report" in result
    assert len(result["analysis_report"]) > 0
    # Phải có các thuật ngữ định lượng của CRO
    assert any(term in result["analysis_report"].lower() for term in ["rủi ro", "score", "phái sinh", "thâm hụt", "risk"])


# ======================================================================
# 4. Kiểm thử FinancialOrchestrator Two-Stage Pipeline & Conditional Routing
# ======================================================================
def test_orchestrator_safe_pipeline_execution(db_path, mock_llm):
    orchestrator = FinancialOrchestrator(db_path=db_path, llm=mock_llm)
    final_state = orchestrator.run("Xem doanh thu trong table_0")

    # Kiểm tra trạng thái hoàn chỉnh qua cả 2 Stage (SQL + Python Quant) và CRO
    assert final_state["error"] is None
    assert "SELECT" in final_state["sql_query"]
    assert isinstance(final_state["sql_result"], pd.DataFrame)
    assert len(final_state["sql_result"]) > 0
    assert final_state["python_code"] is not None
    assert final_state["quant_result"] == 42.0
    assert final_state["analysis_report"] is not None
    assert "🚨 CẢNH BÁO NGUY HIỂM" not in final_state["analysis_report"]

    # Kiểm tra Latency Tracing đã đo lường từng node
    trace = final_state.get("latency_trace")
    assert trace is not None
    assert "sql_generator_ms" in trace
    assert "sql_executor_ms" in trace
    assert "python_quant_ms" in trace
    assert "python_sandbox_ms" in trace
    assert "risk_analyst_ms" in trace
    assert "total_latency_ms" in trace


def test_orchestrator_quant_only_mode(db_path, mock_llm):
    orchestrator = FinancialOrchestrator(db_path=db_path, llm=mock_llm)
    final_state = orchestrator.run("Xem doanh thu trong table_0", quant_only=True)

    assert final_state["error"] is None
    assert final_state["quant_result"] == 42.0
    # quant_only kết thúc sau Agent 2, không chạy Agent 3
    assert final_state["analysis_report"] is None


def test_orchestrator_conditional_edge_security_circuit_breaker(db_path, mock_llm):
    orchestrator = FinancialOrchestrator(db_path=db_path, llm=mock_llm)
    final_state = orchestrator.run("Yêu cầu xóa cơ sở dữ liệu table_0")

    # Kiểm tra Circuit Breaker ngắt luồng và kích hoạt Security Alert
    assert final_state["error"] is not None
    assert "SecurityViolationError" in final_state["error"]
    assert final_state["sql_result"] is None
    assert "🚨 **CẢNH BÁO NGUY HIỂM" in final_state["analysis_report"]
    assert "CIRCUIT BREAKER" in final_state["analysis_report"]
    assert "100% dữ liệu tài chính được bảo vệ" in final_state["analysis_report"]
