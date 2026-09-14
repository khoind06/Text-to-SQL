"""
financial_planner.py
Enterprise-Grade SQL Generator Agent (Natural Language to SQL Translation).

Follows strict Separation of Concerns:
- Dedicated SOLELY to translating natural language inquiries into schema-compliant SQL.
- Extracts dynamic database schema via LangChain SQLDatabase.
- DOES NOT execute SQL queries against the database (delegated to SQLExecutionEngine).
"""

import re
from typing import Any, Dict, List, Optional
from loguru import logger

from src.text_to_sql.sql_generator import NL2SQLGenerator


class FinancialPlannerAgent:
    """
    Tác tử Lập kế hoạch Truy vấn Tài chính (SQL Generator Agent):
    - Tuân thủ nghiêm ngặt nguyên lý Đơn nhiệm (Single Responsibility Principle - SRP).
    - Nhiệm vụ DUY NHẤT: Tiếp nhận câu hỏi tự nhiên và chuyển dịch thành câu truy vấn SQL chuẩn xác.
    - TUYỆT ĐỐI KHÔNG can thiệp thực thi CSDL (Thực thi thuộc về SQLEngine độc lập).
    """

    def __init__(
        self,
        db_path: str = "data/database/finance.db",
        nl2sql_generator: Optional[NL2SQLGenerator] = None,
        llm: Optional[Any] = None,
    ):
        """
        Khởi tạo FinancialPlannerAgent (SQL Generator).

        Args:
            db_path: Đường dẫn CSDL SQLite.
            nl2sql_generator: Instance NL2SQLGenerator (nếu đã có).
            llm: Instance ChatModel.
        """
        self.db_path = db_path
        self.nl2sql = nl2sql_generator or NL2SQLGenerator(
            db_path=db_path,
            llm=llm
        )
        logger.info("[FinancialPlannerAgent (SQL Generator)] Khởi tạo hoàn tất theo chuẩn Single Responsibility.")

    def run_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph Node function cho SQL Generator Agent.

        Args:
            state: Dictionary trạng thái đồ thị (chứa key 'question').

        Returns:
            Dictionary cập nhật trạng thái với key 'sql_query' (hoặc 'error').
        """
        question = state.get("question", "")
        logger.info(f"[SQL Generator Node] Tiếp nhận câu hỏi: '{question}'")

        if not question:
            return {
                "sql_query": "",
                "error": "Câu hỏi đầu vào không được để trống."
            }

        # 1. Xác định đường dẫn CSDL cho Spider hoặc database mặc định
        db_id = state.get("db_id")
        db_path = state.get("db_path")
        if db_id and not db_path:
            db_path = f"data/spider/database/{db_id}/{db_id}.sqlite"

        feedback = state.get("correction_feedback")
        retry_count = state.get("retry_count", 0)

        try:
            # 2. Tạo prompt chứa Lược đồ đa bảng Spider
            prompt_messages = self.nl2sql.create_sql_prompt(question, db_path=db_path)

            # Phản chiếu lỗi thực thi (Self-Correction Reflection) nếu đây là lượt sửa lại
            if feedback and retry_count > 0:
                logger.warning(f"[SQL Generator Node] Kích hoạt Self-Correction (Lần {retry_count}): Áp dụng phản hồi thực thi.")
                from langchain_core.messages import HumanMessage
                correction_prompt = (
                    f"⚠️ CÂU LỆNH SQL TRƯỚC ĐÓ BỊ LỖI THỰC THI TRÊN CƠ SỞ DỮ LIỆU:\n"
                    f"Câu lệnh trước: {state.get('sql_query')}\n"
                    f"Thông báo lỗi chi tiết từ SQLite Engine: {feedback}\n\n"
                    f"YÊU CẦU: Hãy đọc kỹ lỗi trên, đối soát lại chính xác tên cột và kiểu dữ liệu trong schema, "
                    f"và viết lại một câu lệnh SQL duy nhất đã được sửa lỗi hoàn toàn. CHỈ TRẢ VỀ MÃ SQL."
                )
                prompt_messages.append(HumanMessage(content=correction_prompt))

            # 3. Gọi LLM để sinh mã SQL
            response = self.nl2sql.llm.invoke(prompt_messages)
            raw_content = response.content if hasattr(response, "content") else str(response)

            # 4. Làm sạch mã SQL (bóc tách markdown fences)
            clean_sql = self.nl2sql.clean_sql_output(raw_content)
            logger.info(f"[SQL Generator Node] Đã sinh câu lệnh SQL: {clean_sql}")

            return {
                "sql_query": clean_sql,
                "error": None
            }

        except Exception as e:
            err_msg = f"Lỗi trong quá trình sinh câu truy vấn SQL: {str(e)}"
            logger.error(f"[SQL Generator Node] {err_msg}")
            return {
                "sql_query": "",
                "error": err_msg
            }
