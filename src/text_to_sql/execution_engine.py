"""
execution_engine.py
Enterprise-Grade Safe SQL Execution Engine with Defense-in-Depth.

Enforces a 2-layer defense model:
1. Application Layer: RiskGuardrails AST syntax inspection (Zero-Tolerance DDL/DML).
2. Database / OS Layer: SQLite URI locked strictly to Read-Only mode (?mode=ro&uri=true).
   Even if AST validation is somehow bypassed, the underlying SQLite C-Engine
   will reject any write/drop/alter attempt with an OperationalError.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from loguru import logger

from src.text_to_sql.risk_guardrails import RiskGuardrails, SecurityViolationError


class SQLEngine:
    """
    Cơ chế thực thi truy vấn SQL an toàn cấp Enterprise (Defense-in-Depth).
    - Tầng 1 (Guardrails): Kiểm duyệt AST trước khi câu lệnh chạm tới DB.
    - Tầng 2 (Engine): CSDL được mount qua URI Read-Only thuần túy (mode=ro).
    """

    def __init__(
        self,
        db_path: str = "data/database/finance.db",
        guardrails: Optional[RiskGuardrails] = None,
        read_only: bool = True,
    ):
        """
        Khởi tạo SQLEngine với phòng thủ 2 lớp.

        Args:
            db_path: Đường dẫn tới file SQLite database.
            guardrails: Instance RiskGuardrails để kiểm duyệt AST.
            read_only: Khóa kết nối ở chế độ Read-Only (?mode=ro&uri=true).
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            logger.warning(f"[SQLEngine] Database file not found at: {self.db_path}. Verify path.")

        # Cấu hình Defense-in-Depth URI
        resolved_path = self.db_path.resolve().as_posix()
        if read_only:
            # Khóa cứng Read-Only ở tầng C-engine của SQLite
            self.db_uri = f"sqlite:///file:{resolved_path}?mode=ro&uri=true"
        else:
            self.db_uri = f"sqlite:///{resolved_path}"

        self.engine = create_engine(self.db_uri, echo=False)
        self.read_only = read_only
        self._engine_cache: Dict[str, Any] = {resolved_path: self.engine}
        self.guardrails = guardrails or RiskGuardrails()
        logger.info(
            f"[SQLEngine] Initialized with DB: {self.db_path} | "
            f"Mode: {'READ-ONLY (Defense-in-Depth)' if read_only else 'READ-WRITE'}"
        )

    def execute(self, query: str, db_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Thực thi câu truy vấn SQL qua 2 tầng bảo vệ an ninh nghiêm ngặt.

        Args:
            query: Câu truy vấn SQL dạng chuỗi.
            db_path: Đường dẫn CSDL tùy chọn (nếu khác CSDL mặc định).

        Returns:
            pandas.DataFrame chứa kết quả truy vấn.

        Raises:
            SecurityViolationError: Nếu câu lệnh bị từ chối bởi RiskGuardrails hoặc SQLite C-engine.
        """
        # Tầng 1: Kiểm duyệt AST thông qua Risk Guardrails
        is_safe, reason = self.guardrails.validate_query(query)
        if not is_safe:
            err_msg = f"[CẢNH BÁO RỦI RO HỆ THỐNG] Truy vấn bị chặn do vi phạm an ninh: {reason} | Query: {query}"
            logger.error(err_msg)
            raise SecurityViolationError(err_msg)

        # Lựa chọn engine theo db_path động
        target_path = Path(db_path) if db_path else self.db_path
        resolved_path = target_path.resolve().as_posix()
        if resolved_path not in self._engine_cache:
            if self.read_only:
                uri = f"sqlite:///file:{resolved_path}?mode=ro&uri=true"
            else:
                uri = f"sqlite:///{resolved_path}"
            self._engine_cache[resolved_path] = create_engine(uri, echo=False)
        target_engine = self._engine_cache[resolved_path]

        # Tầng 2: Thực thi trên kết nối Read-Only
        try:
            with target_engine.connect() as conn:
                df = pd.read_sql(text(query), con=conn)
            logger.info(f"[SQLEngine] Truy vấn thực thi thành công. Trả về {len(df)} dòng dữ liệu.")
            return df
        except Exception as e:
            err_str = str(e)
            if "readonly database" in err_str.lower() or "attempt to write" in err_str.lower():
                sec_msg = (
                    f"[DEFENSE-IN-DEPTH ENGAGED] SQLite C-Engine đã chặn đứng thao tác ghi "
                    f"trên CSDL Read-Only: {err_str}"
                )
                logger.critical(sec_msg)
                raise SecurityViolationError(sec_msg)

            logger.error(f"[SQLEngine] Lỗi thực thi SQL: {err_str}")
            raise

    def get_tables(self) -> List[str]:
        """Lấy danh sách tất cả các bảng hiện có trong database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Lấy thông tin cấu trúc cột của một bảng cụ thể."""
        inspector = inspect(self.engine)
        return inspector.get_columns(table_name)
