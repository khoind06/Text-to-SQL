"""
sql_generator.py
NL2SQLGenerator: Natural Language to SQL generation module integrated with
LangChain SQLDatabase, local LLMs (ChatOllama), and pre-execution Risk Guardrails.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import text
from loguru import logger
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.text_to_sql.risk_guardrails import RiskGuardrails
from src.text_to_sql.execution_engine import SQLEngine, SecurityViolationError


class NL2SQLGenerator:
    """
    Trình sinh SQL từ Ngôn ngữ tự nhiên (Natural Language to SQL Generator).
    - Kết nối cơ sở dữ liệu tài chính thông qua LangChain SQLDatabase.
    - Cấu hình LLM cục bộ (ChatOllama: qwen2.5:7b, llama3.1:8b hoặc Custom LLM).
    - Tự động trích xuất database schema và tạo System Prompt chuyên dụng cho tài chính.
    - Tích hợp 100% rào chắn an ninh RiskGuardrails trước khi thực thi vào SQLEngine.
    """

    DEFAULT_SYSTEM_TEMPLATE = """Bạn là chuyên gia SQL. Dựa vào lược đồ CSDL được cung cấp, hãy viết một câu truy vấn SQL chuẩn xác. Chú ý sử dụng FOREIGN KEY để thực hiện các phép JOIN giữa nhiều bảng. Chỉ trả về mã SQL thuần túy, không giải thích.

DƯỚI ĐÂY LÀ LƯỢC ĐỒ CƠ SỞ DỮ LIỆU (DATABASE SCHEMA):
{schema}

QUY TẮC BẮT BUỘC:
1. Sử dụng chính xác tên bảng và tên cột trong DATABASE SCHEMA được cung cấp.
2. Chú ý sử dụng FOREIGN KEY để thực hiện các phép JOIN giữa nhiều bảng khi câu hỏi yêu cầu dữ liệu từ nhiều bảng.
3. Hỗ trợ sử dụng các hàm tổng hợp như COUNT(), SUM(), AVG(), MIN(), MAX() và GROUP BY, ORDER BY theo đúng yêu cầu câu hỏi.
4. CHỈ TRẢ VỀ CÂU LỆNH SQL THUẦN TÚY.
5. TUYỆT ĐỐI KHÔNG giải thích, KHÔNG thêm lời mở đầu, KHÔNG kết luận.
6. TUYỆT ĐỐI KHÔNG bọc mã trong khối markdown (như ```sql hoặc ```).
7. CHỈ DÙNG câu lệnh SELECT. Tuyệt đối không dùng INSERT, UPDATE, DELETE, DROP, ALTER.
"""

    def __init__(
        self,
        db_path: str = "data/database/finance.db",
        model_name: str = "qwen2.5-coder:7b",
        llm: Optional[BaseChatModel] = None,
        guardrails: Optional[RiskGuardrails] = None,
        sql_engine: Optional[SQLEngine] = None,
        sample_tables: Optional[List[str]] = None,
    ):
        """
        Khởi tạo NL2SQLGenerator.

        Args:
            db_path: Đường dẫn tới file SQLite database.
            model_name: Tên model Ollama cục bộ (mặc định qwen2.5-coder:7b).
            llm: Instance BaseChatModel tùy biến (cho phép mock LLM trong testing).
            guardrails: Instance RiskGuardrails để kiểm duyệt truy vấn.
            sql_engine: Instance SQLEngine để thực thi an toàn.
            sample_tables: Danh sách bảng đưa vào schema prompt mặc định.
        """
        self.db_path = Path(db_path)
        self.db_uri = f"sqlite:///{self.db_path.resolve().as_posix()}"
        self.model_name = model_name
        self.sample_tables = sample_tables

        # 1. Kết nối database qua LangChain SQLDatabase
        try:
            self.db = SQLDatabase.from_uri(
                self.db_uri,
                sample_rows_in_table_info=2,
                include_tables=None,  # Cho phép truy cập toàn bộ bảng
            )
            logger.info(f"[NL2SQL] Connected to SQLDatabase at: {self.db_path}")
        except Exception as e:
            logger.error(f"[NL2SQL] Error connecting to SQLDatabase: {e}")
            raise

        # 2. Cấu hình LLM
        if llm is not None:
            self.llm = llm
            logger.info("[NL2SQL] Using provided custom/mock LLM.")
        else:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=0.0,
                    num_ctx=4096,  # Tối ưu hóa Context Window giải phóng ~1GB VRAM
                )
                logger.info(f"[NL2SQL] Initialized ChatOllama with model: {self.model_name} (num_ctx=4096)")
            except Exception as e:
                logger.warning(f"[NL2SQL] Could not initialize ChatOllama ({e}). Fallback to None.")
                self.llm = None

        # 3. Khởi tạo RiskGuardrails & SQLEngine
        self.guardrails = guardrails or RiskGuardrails()
        self.sql_engine = sql_engine or SQLEngine(db_path=str(self.db_path), guardrails=self.guardrails)

    def _extract_relevant_rows(self, question: str, table_name: str) -> List[str]:
        """
        Trích xuất các dòng (clean_item_name) phù hợp nhất với câu hỏi bằng RapidFuzz:
        - Lấy toàn bộ danh sách `clean_item_name` phân biệt từ bảng `table_name`.
        - Bóc tách thực thể / tên bảng khỏi câu hỏi và làm sạch chuỗi.
        - So khớp ngữ nghĩa bằng fuzz.token_set_ratio & fuzz.partial_ratio.
        - Trả về top 3 dòng có điểm số tương đồng >= 60.
        """
        try:
            from rapidfuzz import fuzz
            with self.sql_engine.engine.connect() as conn:
                cols = self.sql_engine.get_table_schema(table_name)
                col_names = [c["name"] for c in cols if "name" in c]
                target_col = "clean_item_name" if "clean_item_name" in col_names else ("item_name" if "item_name" in col_names else None)
                if not target_col:
                    return []

                df_items = pd.read_sql(
                    text(f'SELECT DISTINCT "{target_col}" FROM "{table_name}" WHERE "{target_col}" IS NOT NULL AND "{target_col}" != "";'),
                    con=conn
                )
                if df_items.empty:
                    return []
                items = [str(x).strip() for x in df_items[target_col].tolist() if str(x).strip()]

                # Bóc tách tên bảng và chuẩn hóa chuỗi câu hỏi
                q_clean = re.sub(r"\btable_\d+\b", "", question, flags=re.IGNORECASE)
                q_clean = " ".join(re.findall(r"[a-zA-Z0-9_]+", q_clean.lower()))

                scored_items = []
                for it in items:
                    it_clean = " ".join(re.findall(r"[a-zA-Z0-9_]+", it.lower()))
                    if not it_clean:
                        continue
                    score = max(fuzz.token_set_ratio(q_clean, it_clean), fuzz.partial_ratio(it_clean, q_clean))
                    if score >= 60.0:
                        scored_items.append((it, score))

                scored_items.sort(key=lambda x: x[1], reverse=True)
                top_3 = [x[0] for x in scored_items[:3]]
                return top_3
        except Exception as e:
            logger.debug(f"[NL2SQL] Lỗi khi trích xuất row candidates cho {table_name}: {e}")
            return []

    def _get_distinct_categories_and_linking(
        self,
        question: str,
        table_name: str
    ) -> Tuple[List[str], str]:
        """
        Thực hiện Semantic Column Linking, Semantic Row Grounding (RapidFuzz) và Categorical Data Injection.
        - Tìm cột khớp nhất với câu hỏi của người dùng.
        - Trích xuất Row Candidates khớp ngữ nghĩa nhất từ cột clean_item_name.
        - Truy vấn SELECT DISTINCT col LIMIT 15 cho các cột TEXT/VARCHAR.
        """
        try:
            from rapidfuzz import fuzz
            cols = self.sql_engine.get_table_schema(table_name)
            col_names = [c["name"] for c in cols if "name" in c]

            # RapidFuzz so khớp tên cột
            q_clean = " ".join(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
            matched = []
            for col in col_names:
                col_clean = col.lower().replace("_", " ")
                score = max(fuzz.token_set_ratio(q_clean, col_clean), fuzz.partial_ratio(col_clean, q_clean))
                if score >= 60.0:
                    matched.append((col, score))
            matched.sort(key=lambda x: x[1], reverse=True)
            recommended_cols = {m[0] for m in matched[:4]}

            # RapidFuzz Semantic Row Grounding
            row_candidates = self._extract_relevant_rows(question, table_name)

            # Trích xuất dữ liệu phân loại mẫu (Categorical Data Injection)
            schema_lines = [f"\n📌 CHI TIẾT CỘT & GỢI Ý DÒNG PHÙ HỢP CHO '{table_name}':"]
            if row_candidates:
                schema_lines.append(f"  ★ GỢI Ý CÁC DÒNG PHÙ HỢP NHẤT (ROW CANDIDATES): {row_candidates}")

            with self.sql_engine.engine.connect() as conn:
                for c in cols:
                    name = c.get("name", "")
                    col_type = str(c.get("type", "TEXT")).upper()
                    rec_tag = " [★ CỘT LIÊN QUAN TRỌNG TÂM]" if name in recommended_cols else ""
                    
                    sample_info = ""
                    if any(t in col_type for t in ["TEXT", "CHAR", "VARCHAR", "STRING"]):
                        try:
                            # Lấy tối đa 15 giá trị mẫu thực tế
                            sample_df = pd.read_sql(
                                text(f'SELECT DISTINCT "{name}" FROM "{table_name}" WHERE "{name}" IS NOT NULL LIMIT 15;'),
                                con=conn
                            )
                            vals = [str(v).strip() for v in sample_df[name].tolist() if str(v).strip()]
                            if vals:
                                sample_info = f" | Giá trị thực tế có trong DB: {vals}"
                        except Exception:
                            pass

                    schema_lines.append(f"  - {name} ({col_type}){rec_tag}{sample_info}")

            return list(recommended_cols), "\n".join(schema_lines)
        except Exception as e:
            logger.debug(f"[NL2SQL] Bỏ qua Column Linking & Categorical Injection: {e}")
            return [], ""

    # Bộ dữ liệu Abstract Meta-Few-Shot chuẩn mực (Flat SQL Retrieval)
    ABSTRACT_FEW_SHOT_PATTERNS = [
        {
            "category": "Portion / Ratio Calculation",
            "question": "what percentage of total operating expenses was research and development?",
            "sql": "SELECT clean_item_name, metric_value, is_total_row FROM financial_data WHERE clean_item_name LIKE '%research and development%' OR is_total_row = 1;"
        },
        {
            "category": "Year-over-Year Growth Calculation",
            "question": "what was the growth rate of net revenue from fiscal year 2022 to 2023?",
            "sql": "SELECT clean_item_name, fiscal_year_2022, fiscal_year_2023 FROM financial_data WHERE clean_item_name LIKE '%net revenue%';"
        },
        {
            "category": "Data Sanitization & Net Value Calculation",
            "question": "what was the net operating cash flow after deducting capital expenditures?",
            "sql": "SELECT clean_item_name, metric_value FROM financial_data WHERE clean_item_name LIKE '%operating cash flow%' OR clean_item_name LIKE '%capital expenditure%';"
        },
        {
            "category": "Metric Variance Calculation",
            "question": "what was the difference between gross profit and total operating expenses?",
            "sql": "SELECT clean_item_name, metric_value FROM financial_data WHERE clean_item_name LIKE '%gross profit%' OR clean_item_name LIKE '%operating expense%';"
        }
    ]

    def _retrieve_few_shot_examples(self, question: str, k: int = 3) -> str:
        """
        Trích xuất k ví dụ Abstract Meta-Few-Shot tương đồng ngữ nghĩa nhất với câu hỏi người dùng.
        """
        try:
            from rapidfuzz import fuzz
            scored = []
            q_lower = question.lower()
            for item in self.ABSTRACT_FEW_SHOT_PATTERNS:
                score = fuzz.token_set_ratio(q_lower, item["question"].lower())
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_matches = [item for _, item in scored[:k]]

            lines = [
                "\n📌 CÁC MẪU LOGIC TRUY VẤN TÀI CHÍNH TRỪU TƯỢNG (ABSTRACT META-FEW-SHOT EXAMPLES):",
                "⚠️ LƯU Ý BẮT BUỘC: Bảng `financial_data` và các cột `metric_name`, `fiscal_year`, `metric_value` bên dưới chỉ là VÍ DỤ MINH HỌA LOGIC TOÁN HỌC.",
                "Khi sinh câu lệnh SQL thực tế, bạn BẮT BUỘC PHẢI ÁNH XẠ (MAP) sang CHÍNH XÁC tên bảng và tên cột thực tế trong LƯỢC ĐỒ (SCHEMA) ở trên!\n"
            ]
            for idx, ex in enumerate(top_matches, 1):
                lines.append(f"Mẫu {idx} [{ex.get('category', 'Financial Logic')}]:")
                lines.append(f"  Câu hỏi: \"{ex['question']}\"")
                lines.append(f"  Logic SQL mẫu: {ex['sql']}")
            lines.append("")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"[NL2SQL] Bỏ qua Few-Shot RAG: {e}")
            return ""

    @staticmethod
    def extract_spider_schema(db_path: Union[str, Path]) -> str:
        """
        Trích xuất toàn bộ câu lệnh CREATE TABLE đầy đủ từ sqlite_master,
        bao gồm các ràng buộc PRIMARY KEY và FOREIGN KEY rõ ràng cho các bảng trong CSDL.
        """
        import sqlite3
        path = Path(db_path)
        if not path.exists():
            return ""

        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
        tables = cur.fetchall()

        schema_parts = []
        for table_name, create_sql in tables:
            if not create_sql:
                continue

            # Lấy tối đa 2 dòng mẫu đại diện để mô hình thấy định dạng dữ liệu thực tế
            sample_rows_str = ""
            try:
                cur.execute(f'SELECT * FROM "{table_name}" LIMIT 2;')
                rows = cur.fetchall()
                if rows:
                    col_names = [d[0] for d in cur.description]
                    sample_rows_str = f"\n/* 2 dòng mẫu từ {table_name}:\n{col_names}\n"
                    for r in rows:
                        sample_rows_str += f"{list(r)}\n"
                    sample_rows_str += "*/"
            except Exception:
                pass

            schema_parts.append(f"{create_sql.strip()};{sample_rows_str}")

        conn.close()
        return "\n\n".join(schema_parts)

    def create_sql_prompt(
        self,
        question: str,
        db_path: Optional[Union[str, Path]] = None,
        table_names: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Tạo prompt SQL đa bảng chuyên biệt cho Spider dựa trên schema và ràng buộc khóa.
        """
        target_db_path = db_path or self.db_path
        if target_db_path and Path(target_db_path).exists():
            schema_info = self.extract_spider_schema(target_db_path)
        else:
            try:
                schema_info = self.db.get_table_info(table_names=table_names)
            except Exception:
                schema_info = ""

        system_content = self.DEFAULT_SYSTEM_TEMPLATE.format(schema=schema_info)
        user_content = f"Câu hỏi: {question}\n\nSQL:"

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

    def clean_sql_output(self, raw_output: str) -> str:
        """
        Làm sạch output từ LLM: loại bỏ markdown blocks, khoảng trắng thừa,
        đảm bảo trả về câu lệnh SQL thuần túy.
        """
        cleaned = raw_output.strip()

        # Loại bỏ markdown code fences ```sql ... ``` hoặc ``` ... ```
        if "```" in cleaned:
            match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
            else:
                cleaned = re.sub(r"```(?:sql)?", "", cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.replace("```", "").strip()

        # Loại bỏ các từ thừa mở đầu như "SQL:", "Output:", "Answer:" nếu LLM thêm vào
        cleaned = re.sub(r"^(?:SQL|Query|Output|Answer|Truy vấn):\s*", "", cleaned, flags=re.IGNORECASE).strip()

        return cleaned

    def generate_and_execute(
        self,
        question: str,
        table_names: Optional[List[str]] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Quy trình xử lý hoàn chỉnh từ Ngôn ngữ tự nhiên -> SQL -> Kiểm duyệt -> Thực thi:
        - B1: Đưa question vào LLM để sinh SQL string.
        - B2: Gọi RiskGuardrails để parse AST và kiểm duyệt câu lệnh.
        - B3: Nếu an toàn, gọi SQLEngine.execute() lấy Pandas DataFrame.
              Nếu vi phạm, trả về lỗi bảo mật (hoặc ném SecurityViolationError nếu raise_on_error=True).

        Args:
            question: Câu hỏi phân tích tài chính.
            table_names: Danh sách bảng mục tiêu cần cung cấp schema cho LLM.
            raise_on_error: Nếu True, ném SecurityViolationError khi bị chặn.

        Returns:
            Dict chứa status, question, sql_query, data, row_count, reason.
        """
        logger.info(f"[NL2SQL] Received question: '{question}'")

        # B1: Đưa question vào LLM để sinh SQL string
        if self.llm is None:
            error_msg = "LLM chưa được khởi tạo. Cần truyền LLM hoặc cài đặt Ollama."
            logger.error(f"[NL2SQL] {error_msg}")
            if raise_on_error:
                raise RuntimeError(error_msg)
            return {
                "status": "error",
                "question": question,
                "sql_query": "",
                "data": None,
                "error": error_msg,
            }

        prompt_messages = self.create_sql_prompt(question, table_names=table_names)
        try:
            response = self.llm.invoke(prompt_messages)
            raw_content = response.content if hasattr(response, "content") else str(response)
            generated_sql = self.clean_sql_output(raw_content)
            logger.info(f"[NL2SQL] Generated SQL: {generated_sql}")
        except Exception as e:
            error_msg = f"Lỗi trong quá trình LLM sinh câu truy vấn: {str(e)}"
            logger.error(f"[NL2SQL] {error_msg}")
            if raise_on_error:
                raise
            return {
                "status": "llm_error",
                "question": question,
                "sql_query": "",
                "data": None,
                "error": error_msg,
            }

        # B2: Gọi RiskGuardrails để parse AST và kiểm duyệt câu lệnh
        is_safe, reason = self.guardrails.validate_query(generated_sql)
        if not is_safe:
            sec_msg = f"[RỦI RO AN NINH] Câu truy vấn bị chặn bởi RiskGuardrails: {reason} | Query: {generated_sql}"
            logger.warning(f"[NL2SQL] {sec_msg}")
            if raise_on_error:
                raise SecurityViolationError(sec_msg)
            return {
                "status": "security_violation",
                "question": question,
                "sql_query": generated_sql,
                "data": None,
                "row_count": 0,
                "is_safe": False,
                "reason": reason,
                "error": sec_msg,
            }

        # B3: Nếu an toàn, gọi SQLEngine.execute() để lấy dữ liệu dạng Pandas DataFrame
        try:
            df_result = self.sql_engine.execute(generated_sql)
            logger.success(f"[NL2SQL] Successfully executed query. Returned {len(df_result)} rows.")
            return {
                "status": "success",
                "question": question,
                "sql_query": generated_sql,
                "data": df_result,
                "row_count": len(df_result),
                "is_safe": True,
                "reason": "Truy vấn an toàn và thực thi thành công.",
                "error": None,
            }
        except Exception as e:
            exec_msg = f"Lỗi khi thực thi câu truy vấn SQL: {str(e)}"
            logger.error(f"[NL2SQL] {exec_msg}")
            if raise_on_error:
                raise
            return {
                "status": "execution_error",
                "question": question,
                "sql_query": generated_sql,
                "data": None,
                "row_count": 0,
                "is_safe": True,
                "error": exec_msg,
            }
