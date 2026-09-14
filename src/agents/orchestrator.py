"""
orchestrator.py
Single-Agent SQL Orchestration Engine using LangGraph for Relational Databases (Spider Benchmark).

Topology:
START -> sql_generator -> sql_executor -> [Conditional Edge: Safety & Execution Check]
                               |---> If Security Violation -> security_alert -> END
                               |---> If SQL Syntax/Schema Error & retries < 3 -> self_correction -> sql_generator
                               |---> If Success or max retries -> END
"""

import sys
import time
from pathlib import Path

# Cấu hình UTF-8 cho Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo thư mục gốc dự án luôn có trong sys.path khi chạy script độc lập
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, Optional, TypedDict, Union
import pandas as pd
from loguru import logger
from langgraph.graph import StateGraph, START, END

from src.agents.financial_planner import FinancialPlannerAgent
from src.text_to_sql.risk_guardrails import RiskGuardrails
from src.text_to_sql.execution_engine import SQLEngine, SecurityViolationError


# ======================================================================
# 1. Định nghĩa trạng thái của đồ thị (Single-Agent State)
# ======================================================================
class AgentState(TypedDict):
    """
    Trạng thái được truyền qua các node trong đồ thị LangGraph Single-Agent SQL:
    - question: Câu hỏi ngôn ngữ tự nhiên từ người dùng.
    - db_id: Tên định danh CSDL trong Spider (ví dụ: concert_singer).
    - db_path: Đường dẫn vật lý tới file SQLite tương ứng.
    - sql_query: Câu lệnh SQL do SQL Generator Agent sinh ra.
    - sql_result: Dữ liệu bảng do Execution Engine trích xuất (pd.DataFrame).
    - error: Cảnh báo vi phạm an ninh hoặc lỗi thực thi cú pháp (nếu có).
    - latency_trace: Nhật ký đo lường thời gian phản hồi (ms) của từng node.
    - retry_count: Số lần đã kích hoạt Self-Correction retry cho SQL.
    - correction_feedback: Phản hồi lỗi thực thi trả về cho SQL Generator để tự sửa.
    """
    question: str
    db_id: Optional[str]
    db_path: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[Any]
    error: Optional[str]
    latency_trace: Optional[Dict[str, float]]
    retry_count: Optional[int]
    correction_feedback: Optional[str]


# ======================================================================
# 2. Class Điều phối trung tâm (FinancialOrchestrator)
# ======================================================================
class FinancialOrchestrator:
    """
    Bộ điều phối Tác tử SQL Đơn nhiệm (Single-Agent SQL Orchestrator):
    - Agent: SQL Generator (Dịch câu hỏi tự nhiên sang SQL đa bảng theo schema quan hệ).
    - Execution Engine: Kiểm duyệt AST qua RiskGuardrails và chạy CSDL an toàn (Read-Only).
    - Self-Correction: Tự động phản hồi lỗi cú pháp/schema để Agent tự sửa tối đa 3 lần.
    - Circuit Breaker: Tự động ngắt luồng an toàn khi phát hiện tấn công dữ liệu.
    """

    def __init__(
        self,
        db_path: str = "data/database/finance.db",
        sql_model: str = "qwen2.5-coder:7b",
        llm: Optional[Any] = None,
        planner_agent: Optional[FinancialPlannerAgent] = None,
        guardrails: Optional[RiskGuardrails] = None,
        sql_engine: Optional[SQLEngine] = None,
    ):
        """
        Khởi tạo Single-Agent SQL Orchestrator.

        Args:
            db_path: Đường dẫn CSDL SQLite mặc định.
            sql_model: Mô hình cho SQL Generator (mặc định qwen2.5-coder:7b).
            llm: Instance ChatModel dùng chung (nếu muốn override).
            planner_agent: Instance FinancialPlannerAgent (SQL Generator).
            guardrails: Instance RiskGuardrails.
            sql_engine: Instance SQLEngine.
        """
        self.db_path = db_path
        self.llm = llm

        # SQL Generator Agent
        if planner_agent:
            self.planner = planner_agent
        elif llm:
            self.planner = FinancialPlannerAgent(db_path=db_path, llm=llm)
        else:
            from src.text_to_sql.sql_generator import NL2SQLGenerator
            sql_gen = NL2SQLGenerator(db_path=db_path, model_name=sql_model)
            self.planner = FinancialPlannerAgent(db_path=db_path, nl2sql_generator=sql_gen)

        self.guardrails = guardrails or RiskGuardrails()
        self.sql_engine = sql_engine or SQLEngine(db_path=db_path, guardrails=self.guardrails)

        # Xây dựng đồ thị StateGraph Single-Agent
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        logger.info("[FinancialOrchestrator] Single-Agent SQL StateGraph đã biên dịch thành công cho Spider.")

    # ── NODE 1: SQL GENERATOR AGENT ────────────────────────────────────
    def _sql_generator_node(self, state: AgentState) -> Dict[str, Any]:
        """Node dịch câu hỏi thành câu lệnh SQL dựa trên Lược đồ quan hệ đa bảng."""
        t0 = time.perf_counter()
        trace = dict(state.get("latency_trace") or {})

        planner_result = self.planner.run_node(state)
        trace["sql_generator_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "sql_query": planner_result.get("sql_query", ""),
            "error": planner_result.get("error"),
            "latency_trace": trace,
        }

    # ── NODE 2: EXECUTION ENGINE & GUARDRAIL NODE ──────────────────────
    def _sql_executor_node(self, state: AgentState) -> Dict[str, Any]:
        """Node kiểm duyệt AST qua RiskGuardrails và thực thi an toàn trên CSDL."""
        t0 = time.perf_counter()
        trace = dict(state.get("latency_trace") or {})

        sql_query = state.get("sql_query", "")
        db_path = state.get("db_path") or self.db_path

        if not sql_query or state.get("error"):
            trace["sql_executor_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return {"sql_result": None, "latency_trace": trace}

        # 1. Kiểm duyệt an ninh AST
        is_safe, reason = self.guardrails.validate_query(sql_query)
        if not is_safe:
            sec_err = f"SecurityViolationError: [CẢNH BÁO AN NINH AST] {reason} | Query: {sql_query}"
            logger.warning(f"[SQLExecutor Node] Chặn đứng vi phạm an ninh: {sec_err}")
            trace["sql_executor_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "sql_result": None,
                "error": sec_err,
                "latency_trace": trace,
            }

        # 2. Thực thi an toàn trên CSDL tương ứng
        try:
            df = self.sql_engine.execute(sql_query, db_path=db_path)
            logger.success(f"[SQLExecutor Node] Thực thi SQL thành công trên DB '{db_path}'. Thu được {len(df)} dòng dữ liệu.")
            trace["sql_executor_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "sql_result": df,
                "error": None,
                "latency_trace": trace,
            }
        except SecurityViolationError as sve:
            logger.warning(f"[SQLExecutor Node] Bắt giữ SecurityViolationError: {sve}")
            trace["sql_executor_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "sql_result": None,
                "error": f"SecurityViolationError: {str(sve)}",
                "latency_trace": trace,
            }
        except Exception as e:
            err_msg = str(e)
            logger.error(f"[SQLExecutor Node] Lỗi thực thi SQL: {err_msg}")
            trace["sql_executor_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "sql_result": None,
                "error": f"ExecutionError: {err_msg}",
                "latency_trace": trace,
            }

    # ── NODE 3: SECURITY ALERT CIRCUIT BREAKER ─────────────────────────
    def _security_alert_node(self, state: AgentState) -> Dict[str, Any]:
        """Node ngắt mạch an toàn (Circuit Breaker) và xuất cảnh báo khẩn cấp."""
        t0 = time.perf_counter()
        trace = dict(state.get("latency_trace") or {})

        error_msg = state.get("error", "Phát hiện hành vi vi phạm an ninh nghiêm trọng.")
        logger.critical(f"[Security Alert Node] Kích hoạt rào chắn phòng thủ: {error_msg}")

        trace["security_alert_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "sql_result": None,
            "latency_trace": trace,
        }

    # ── NODE 4: SQL SELF-CORRECTION REFLECTION NODE ────────────────────
    def _self_correction_node(self, state: AgentState) -> Dict[str, Any]:
        """Node đóng gói phản hồi lỗi thực thi để chuyển ngược về SQL Generator."""
        retry_count = state.get("retry_count", 0) + 1
        error_msg = state.get("error", "Lỗi cú pháp SQL")
        logger.warning(f"[Self-Correction Node] Đóng gói lỗi thực thi SQL để phản chiếu (Lần {retry_count}): {error_msg}")

        return {
            "retry_count": retry_count,
            "correction_feedback": str(error_msg),
            "error": None,
        }

    # ── CONDITIONAL ROUTING ───────────────────────────────────────────
    def _route_after_executor(self, state: AgentState) -> str:
        """
        Rẽ nhánh có điều kiện sau bước SQL Execution:
        - Nếu vi phạm an ninh (SecurityViolationError) -> Chặn đứng ngay, rẽ sang 'security_alert'.
        - Nếu lỗi cú pháp SQL hoặc schema error:
          - Nếu retry_count < 3 -> Rẽ sang 'self_correction' để Agent tự sửa.
          - Nếu đã quá 3 lần -> Rẽ sang 'end'.
        - Nếu an toàn và thực thi thành công -> Rẽ sang 'end'.
        """
        error = state.get("error")
        if error is not None and len(str(error).strip()) > 0:
            error_str = str(error)
            # 1. Vi phạm bảo mật: Ngắt ngay lập tức, không cho retry
            if "SecurityViolationError" in error_str or "an ninh AST" in error_str:
                logger.warning("[Orchestrator Route] Phát hiện vi phạm an ninh nghiêm trọng. Bẻ hướng luồng sang 'security_alert'.")
                return "security_alert"

            # 2. Lỗi thực thi cú pháp hoặc schema: Kích hoạt Self-Correction nếu còn lượt
            retry_count = state.get("retry_count", 0)
            max_retries = 3
            if retry_count < max_retries:
                logger.info(f"[Orchestrator Route] Phát hiện lỗi thực thi. Kích hoạt SQL Self-Correction (Lần {retry_count + 1}/{max_retries}).")
                return "self_correction"
            else:
                logger.warning(f"[Orchestrator Route] Đã hết lượt Self-Correction ({max_retries} lần). Kết thúc.")
                return "end"

        return "end"

    # ── ĐÓNG GÓI ĐỒ THỊ LANGGRAPH ──────────────────────────────────────
    def _build_graph(self) -> StateGraph:
        """Xây dựng đồ thị trạng thái Single-Agent SQL hoàn chỉnh."""
        builder = StateGraph(AgentState)

        # 1. Thêm các Node
        builder.add_node("sql_generator", self._sql_generator_node)
        builder.add_node("sql_executor", self._sql_executor_node)
        builder.add_node("self_correction", self._self_correction_node)
        builder.add_node("security_alert", self._security_alert_node)

        # 2. Định nghĩa các Luồng (Edges)
        builder.add_edge(START, "sql_generator")
        builder.add_edge("sql_generator", "sql_executor")

        # Rẽ nhánh có điều kiện sau khi kiểm duyệt và thực thi SQL
        builder.add_conditional_edges(
            "sql_executor",
            self._route_after_executor,
            {
                "security_alert": "security_alert",
                "self_correction": "self_correction",
                "end": END,
            }
        )

        # Self-Correction phản hồi ngược lại cho SQL Generator để tự sửa
        builder.add_edge("self_correction", "sql_generator")
        builder.add_edge("security_alert", END)

        return builder

    def run(
        self,
        question: str,
        db_id: Optional[str] = None,
        db_path: Optional[str] = None,
        **kwargs
    ) -> AgentState:
        """
        Thực thi toàn bộ luồng Single-Agent với câu hỏi đầu vào.

        Args:
            question: Câu hỏi ngôn ngữ tự nhiên từ người dùng.
            db_id: Tên CSDL Spider (nếu có).
            db_path: Đường dẫn vật lý tới CSDL SQLite (nếu có).

        Returns:
            AgentState cuối cùng chứa câu lệnh sql_query, sql_result và latency trace.
        """
        # Tự động suy diễn db_path nếu có db_id
        if db_id and not db_path:
            db_path = f"data/spider/database/{db_id}/{db_id}.sqlite"

        logger.info(
            f"\n{'='*70}\n"
            f"[Orchestrator] Bắt đầu phiên làm việc Spider: '{question}' (DB: '{db_id or db_path}')\n"
            f"{'='*70}"
        )
        t_start = time.perf_counter()

        initial_state: AgentState = {
            "question": question,
            "db_id": db_id,
            "db_path": db_path,
            "sql_query": None,
            "sql_result": None,
            "error": None,
            "latency_trace": {},
            "retry_count": 0,
            "correction_feedback": None,
        }

        final_state = self.app.invoke(initial_state)

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        if final_state.get("latency_trace") is not None:
            final_state["latency_trace"]["total_latency_ms"] = total_latency

        has_data = final_state.get('sql_result') is not None and isinstance(final_state.get('sql_result'), pd.DataFrame)
        rows = len(final_state.get('sql_result')) if has_data else 0
        logger.success(
            f"[Orchestrator] Hoàn thành phiên làm việc trong {total_latency} ms. "
            f"SQL: {final_state.get('sql_query')} | Kết quả: {rows} dòng."
        )

        return final_state
