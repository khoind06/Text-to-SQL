"""
python_quant_engine.py
Python Quant Engine & Zero-Trust Sandbox (Agent 2):
Thực thi các phép tính toán tài chính đa bước từ DataFrame bằng mã Python/Pandas
được kiểm duyệt an ninh qua Abstract Syntax Tree (AST) Sandbox.
"""

import ast
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from loguru import logger
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


class SandboxSecurityError(Exception):
    """Ném ra khi mã Python do LLM sinh ra vi phạm chính sách an ninh AST."""
    pass


class PythonQuantSandbox:
    """
    Hộp cát an ninh Zero-Trust cho phép thực thi mã Python định lượng một cách biệt lập.
    - Duyệt AST phát hiện và chặn đứng mọi import nguy hại (os, sys, subprocess, shutil...).
    - Vô hiệu hóa các built-in nguy hiểm (eval, exec, open, __import__).
    - Cung cấp môi trường thực thi khép kín chỉ chứa pandas, numpy, math.
    """

    FORBIDDEN_MODULES: Set[str] = {
        "os", "sys", "subprocess", "shutil", "pathlib", "socket", "http",
        "urllib", "requests", "importlib", "builtins", "posix", "nt",
        "inspect", "pickle", "ctypes", "multiprocessing", "threading"
    }

    FORBIDDEN_CALLS: Set[str] = {
        "eval", "exec", "compile", "open", "getattr", "setattr",
        "delattr", "__import__", "globals", "locals", "breakpoint"
    }

    @classmethod
    def validate_ast(cls, code_str: str) -> Tuple[bool, str]:
        """
        Kiểm tra an ninh AST của mã nguồn Python trước khi thực thi.
        """
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"Lỗi cú pháp Python SyntaxError: {e}"

        for node in ast.walk(tree):
            # 1. Kiểm tra Import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in cls.FORBIDDEN_MODULES:
                        return False, f"Vi phạm an ninh: Cấm import thư viện hệ thống '{root_mod}'."

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in cls.FORBIDDEN_MODULES:
                        return False, f"Vi phạm an ninh: Cấm import thư viện hệ thống '{root_mod}'."

            # 2. Kiểm tra hàm gọi nguy hiểm
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in cls.FORBIDDEN_CALLS:
                    return False, f"Vi phạm an ninh: Cấm gọi hàm nguy hiểm '{func_name}()'."

        return True, "Code AST an toàn."

    @classmethod
    def execute(cls, code_str: str, df: pd.DataFrame) -> Tuple[Optional[Any], Optional[str]]:
        """
        Thực thi an toàn đoạn mã Python trên DataFrame.
        Kết quả tính toán bắt buộc phải được gán vào biến `result`.

        Returns:
            (result_value, error_message)
        """
        # Bước 1: Thẩm định AST
        is_safe, msg = cls.validate_ast(code_str)
        if not is_safe:
            logger.warning(f"[QuantSandbox] AST Blocked: {msg}")
            return None, msg

        # Bước 2: Chuẩn bị môi trường globals hạn chế
        safe_globals = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "pow": pow,
                "range": range,
                "round": round,
                "set": set,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "print": lambda *args, **kwargs: None,
            },
            "pd": pd,
            "np": np,
            "math": math,
            "df": df.copy(),
        }
        local_vars: Dict[str, Any] = {}

        # Bước 3: Thực thi mã
        try:
            exec(code_str, safe_globals, local_vars)
        except ZeroDivisionError:
            return 0.0, "ZeroDivisionError: Mẫu số bằng 0 (đã trả về 0.0 an toàn)."
        except Exception as e:
            return None, f"RuntimeError trong Sandbox: {type(e).__name__}: {str(e)}"

        # Bước 4: Trích xuất biến `result`
        if "result" not in local_vars:
            # Fallback: tìm biến bất kỳ được tính toán cuối cùng
            candidates = [k for k in local_vars.keys() if not k.startswith("_")]
            if candidates:
                result_val = local_vars[candidates[-1]]
                logger.info(f"[QuantSandbox] Biến 'result' không được gán rõ ràng, fallback sang: {candidates[-1]} = {result_val}")
                return result_val, None
            return None, "Lỗi thực thi: Không tìm thấy biến 'result' trong mã tính toán."

        return local_vars["result"], None


class PythonQuantAgent:
    """
    Tác tử Tính toán Định lượng Python (Agent 2 - Python Quant Engine).
    - Nhận dữ liệu DataFrame sạch từ Agent 1 (SQL Extractor).
    - Sinh mã Python/Pandas ngắn gọn, chính xác để giải toán đa bước.
    - Tự sửa lỗi qua ReAct feedback nếu Sandbox trả về lỗi.
    """

    SYSTEM_PROMPT = """Bạn là một chuyên gia Lập trình Định lượng (Financial Quant Engineer) chuyên viết mã Python/Pandas để phân tích dữ liệu tài chính từ DataFrame `df`.
Nhiệm vụ của bạn: Đọc câu hỏi của người dùng và bảng dữ liệu `df`, viết MỘT đoạn mã Python duy nhất để tính toán ra đáp án chuẩn xác.

3 NGUYÊN TẮC LẬP TRÌNH BẮT BUỘC (CRITICAL CODING GUIDELINES):

1. KỸ THUẬT LỌC PANDAS MỀM DẺO (ROBUST FILTERING):
   - NGHIÊM CẤM sử dụng toán tử so sánh tuyệt đối `==` khi lọc chuỗi (ví dụ: TUYỆT ĐỐI KHÔNG viết `df[df['clean_item_name'] == 'net revenue']`).
   - BẮT BUỘC sử dụng `.str.contains('keyword', case=False, na=False)`.
   - Cú pháp chuẩn: `filtered = df[df['clean_item_name'].str.contains('keyword', case=False, na=False)]`

2. BỘ CÔNG THỨC KẾ TOÁN TIÊU CHUẨN (FINANCIAL FORMULA GROUNDING):
   - TĂNG TRƯỞNG LIÊN KỲ (Growth Rate / YoY / Percentage Change):
     + BẮT BUỘC lấy: `(Năm_Sau - Năm_Trước) / Năm_Trước`.
     + TUYỆT ĐỐI KHÔNG chia cho Năm_Sau! Mẫu số luôn là Năm_Trước (gốc).
   - TỶ TRỌNG (Ratio / Percentage / Portion / Proportion of Total):
     + BẮT BUỘC lấy: `Thành_Phần / Tổng`.
     + Nếu bảng có dòng chứa chữ 'total' (hoặc `is_total_row == 1`), phải ưu tiên bóc giá trị từ dòng 'total' đó làm mẫu số.
     + LUÔN tính dưới dạng số thập phân thuần túy (ví dụ: 15 / 100 = 0.15). TUYỆT ĐỐI KHÔNG nhân 100!
   - THAY ĐỔI RÒNG (Net Change / Difference / Variance):
     + Lấy `Năm_Sau - Năm_Trước` hoặc `Giá_trị_A - Giá_trị_B` tùy ngữ cảnh câu hỏi.
   - CÂU HỎI SO SÁNH BOOLEAN (Yes / No):
     + Nếu câu hỏi bắt đầu bằng "did", "was", "is", "does" (hỏi xem chỉ số A có lớn hơn chỉ số B không):
     + BẮT BUỘC gán `result = 'yes'` nếu điều kiện đúng, hoặc `result = 'no'` nếu sai.

3. XỬ LÝ NGOẠI LỆ & AN TOÀN TRUY CẬP (GRACEFUL FALLBACK & ZERO-DIVISION):
   - Trước khi trích xuất `.values[0]` hoặc `.iloc[0]`, LUÔN kiểm tra DataFrame đã lọc có rỗng không:
     ```python
     filtered = df[df['clean_item_name'].str.contains('keyword', case=False, na=False)]
     val = filtered['col_name'].values[0] if not filtered.empty else 0.0
     ```
   - Chống chia cho 0: `result = numer / denom if denom != 0 else 0.0`

4. ĐỊNH DẠNG ĐẦU RA:
   - Dữ liệu đã có sẵn trong biến `df`. KHÔNG import pandas, KHÔNG đọc file.
   - Đáp án cuối cùng BẮT BUỘC gán vào biến `result`.
   - CHỈ TRẢ VỀ MÃ PYTHON THUẦN TÚY, KHÔNG giải thích, KHÔNG bọc trong markdown fences.

MẪU CODE MINH HỌA CHUẨN:
```python
# Tăng trưởng doanh thu 2014-2015:
f_rev = df[df['clean_item_name'].str.contains('revenue', case=False, na=False)]
v_2015 = f_rev['col_2015'].values[0] if not f_rev.empty else 0.0
v_2014 = f_rev['col_2014'].values[0] if not f_rev.empty else 0.0
result = (v_2015 - v_2014) / v_2014 if v_2014 != 0 else 0.0
```
"""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        llm: Optional[BaseChatModel] = None,
    ):
        self.model_name = model_name
        if llm is not None:
            self.llm = llm
            logger.info("[PythonQuantAgent] Sử dụng Custom/Mock LLM được cung cấp.")
        else:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=0.0,
                    num_ctx=4096,
                )
                logger.info(f"[PythonQuantAgent] Đã khởi tạo ChatOllama với model: {self.model_name}")
            except Exception as e:
                logger.warning(f"[PythonQuantAgent] Không thể khởi tạo ChatOllama ({e}). Fallback to None.")
                self.llm = None

        self.sandbox = PythonQuantSandbox()

    def clean_python_output(self, raw_output: str) -> str:
        """Làm sạch markdown code blocks từ output của LLM."""
        cleaned = raw_output.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:python|py)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
            else:
                cleaned = cleaned.replace("```python", "").replace("```py", "").replace("```", "").strip()

        cleaned = re.sub(r"^(?:Python|Code|Solution):\s*", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def run_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi node Python Quant Agent trong LangGraph:
        - Lấy DataFrame từ `sql_result`.
        - Sinh mã Python tính toán.
        - Gọi Sandbox thực thi và lấy `result`.
        """
        question = state.get("question", "")
        df_result = state.get("sql_result")
        feedback = state.get("python_feedback")
        retry_count = state.get("python_retry_count", 0)

        # 1. Kiểm tra nếu không có dữ liệu đầu vào từ Agent 1
        if df_result is None or not isinstance(df_result, pd.DataFrame) or df_result.empty:
            return {
                "python_code": "",
                "quant_result": None,
                "error": "PythonQuantError: Không có dữ liệu DataFrame hợp lệ từ Agent 1 (SQL trống)."
            }

        if self.llm is None:
            return {
                "python_code": "",
                "quant_result": None,
                "error": "PythonQuantError: LLM chưa được khởi tạo."
            }

        # 2. Chuẩn bị bối cảnh dữ liệu trực quan dạng bảng
        df_table_str = df_result.head(20).to_string(index=False)
        user_prompt = (
            f"Câu hỏi tài chính: {question}\n\n"
            f"DataFrame `df` hiện có {len(df_result)} dòng với các cột: {list(df_result.columns)}\n"
            f"Nội dung bảng `df`:\n{df_table_str}\n\n"
            f"Hãy viết mã Python/Pandas theo đúng 3 nguyên tắc (dùng str.contains, công thức tài chính chuẩn, fallback an toàn) để tính toán câu trả lời và gán vào biến `result`:"
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # Phản chiếu lỗi thực thi (Self-Correction Reflection) nếu đây là lần thử lại
        if feedback and retry_count > 0:
            logger.warning(f"[PythonQuantAgent] Áp dụng phản hồi sửa lỗi Python (Lần {retry_count}): {feedback}")
            correction_msg = (
                f"⚠️ MÃ PYTHON TRƯỚC ĐÓ THỰC THI BỊ LỖI:\n"
                f"Mã trước: {state.get('python_code')}\n"
                f"Chi tiết lỗi từ Python Sandbox: {feedback}\n\n"
                f"YÊU CẦU: Sửa lại mã để không bị lỗi trên, đảm bảo biến `result` nhận giá trị cuối cùng. CHỈ TRẢ VỀ MÃ PYTHON."
            )
            messages.append(HumanMessage(content=correction_msg))

        try:
            resp = self.llm.invoke(messages)
            raw_code = resp.content if hasattr(resp, "content") else str(resp)
            clean_code = self.clean_python_output(raw_code)
            logger.info(f"[PythonQuantAgent] Đã sinh mã Python:\n{clean_code}")

            return {
                "python_code": clean_code,
                "error": None
            }
        except Exception as e:
            err_msg = f"Lỗi sinh mã Python: {str(e)}"
            logger.error(f"[PythonQuantAgent] {err_msg}")
            return {
                "python_code": "",
                "quant_result": None,
                "error": err_msg
            }
