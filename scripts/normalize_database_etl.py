"""
normalize_database_etl.py
Advanced Comprehensive Financial ETL Pipeline V2.
Chuẩn hóa toàn diện cơ sở dữ liệu tài chính SEC 10-K từ FinQA sang SQLite:
1. Extended Numeric Sanitization: Unicode dashes, số âm kế toán $(x), footnotes, tiền tệ, 0.0 vs NULL.
2. Column Metadata & Sanitization: Tránh xung đột số, trùng lặp tên cột, làm sạch ký tự lạ.
3. Feature Engineering cho LLM: Thêm cột `is_total_row` (0 hoặc 1) và `clean_item_name`.
4. Hierarchical / Merged Headers: Xử lý tiêu đề phân tầng / niên độ.
5. Zero-Trust & An toàn: Sao lưu database trước khi chạy, xóa sạch ground truth khỏi DB, VACUUM.
"""

import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class ExtendedNumericSanitizer:
    """Bộ làm sạch và ép kiểu số liệu tài chính chuyên sâu."""

    UNICODE_DASHES = re.compile(r"[\u2212\u2013\u2014]")
    CURRENCY_SYMBOLS = re.compile(r"[\$€£¥%]|(?:CAD|USD|R)\$?", re.IGNORECASE)
    FOOTNOTE_PATTERN = re.compile(
        r"(?:<[^>]+>|\[.*?\]|\([a-zA-Z0-9]+\)|[\*†‡#]).*$"
    )
    EMPTY_PATTERNS = {"", "-", "—", "–", "n/a", "na", "none", "null", "nil"}

    @classmethod
    def parse_cell(cls, val: Any) -> Tuple[Optional[float], bool]:
        """
        Phân tích một ô dữ liệu thành (float_value, is_numeric).
        Hỗ trợ:
        - Số âm kế toán: $(4,614) hoặc (1,234.5) -> -4614.0, -1234.5
        - Dấu gạch Unicode: U+2212 (−), U+2013 (–), U+2014 (—) -> -
        - Footnotes: 12,450 (1), 45.2 (a), 1,098* -> 12450.0, 45.2, 1098.0
        - Tiền tệ & đơn vị: $, €, £, ¥, %, CAD$, USD
        - Ô rỗng hoặc gạch ngang: '-', '—', 'N/A' -> 0.0
        """
        if val is None:
            return 0.0, True

        raw_str = str(val).strip()
        lower_str = raw_str.lower()

        if lower_str in cls.EMPTY_PATTERNS:
            return 0.0, True

        cleaned = cls.UNICODE_DASHES.sub("-", raw_str).strip()

        is_negative = False
        m_neg = re.search(r"^\s*\(?\s*\$?\s*\((.*?)\)\s*$", cleaned) or re.search(
            r"^\s*\$?\s*\((.*?)\)\s*$", cleaned
        )
        if m_neg:
            is_negative = True
            cleaned = m_neg.group(1).strip()
        elif cleaned.startswith("-") or cleaned.startswith("$-"):
            is_negative = True
            cleaned = re.sub(r"^\s*[-$]+\s*", "", cleaned).strip()

        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        cleaned = cls.FOOTNOTE_PATTERN.sub("", cleaned).strip()
        cleaned = cls.CURRENCY_SYMBOLS.sub("", cleaned).strip()
        cleaned = cleaned.replace(",", "").replace(" ", "")

        try:
            num_val = float(cleaned)
            return (-num_val if is_negative else num_val), True
        except (ValueError, TypeError):
            return None, False


class FinancialDataETL:
    """Pipeline ETL Chuẩn hóa Dữ liệu Tài chính cấp Enterprise V2."""

    TOTAL_KEYWORDS = ["total", "sum", "balance as of", "aggregate"]

    def __init__(
        self,
        raw_json_path: str = "FinQA/dataset/test.json",
        db_path: str = "data/database/finance.db",
        backup_path: str = "data/database/finance_backup.db",
    ):
        self.raw_json_path = Path(raw_json_path)
        self.db_path = Path(db_path)
        self.backup_path = Path(backup_path)

    def backup_existing_db(self):
        """Tạo bản sao lưu CSDL hiện tại trước khi thực thi."""
        if self.db_path.exists():
            self.backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.db_path, self.backup_path)
            logger.success(f"[ETL] Đã tạo bản sao lưu CSDL an toàn tại: {self.backup_path}")
        else:
            logger.warning(f"[ETL] Không tìm thấy DB cũ tại {self.db_path}, bỏ qua sao lưu.")

    @classmethod
    def clean_text_label(cls, label: Any) -> str:
        """Chuẩn hóa canonical item name: xóa dấu câu, lowercase, khoảng trắng đơn."""
        if label is None:
            return ""
        t = re.sub(r"<[^>]+>", "", str(label))
        t = re.sub(r"\([a-zA-Z0-9]+\)$", "", t).strip()
        t = re.sub(r"\[.*?\]", "", t).strip()
        t = re.sub(r"[\*†‡#]", "", t).strip()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip().lower()
        return t

    @classmethod
    def check_is_total_row(cls, text_val: str) -> int:
        """Kiểm tra dòng có phải dòng tổng cộng không."""
        lower = str(text_val).lower()
        return 1 if any(kw in lower for kw in cls.TOTAL_KEYWORDS) else 0

    @classmethod
    def sanitize_column_name(cls, raw_name: Any, col_idx: int, existing_cols: set) -> str:
        """
        Chuẩn hóa tên cột hợp chuẩn SQL:
        - Lowercase, thay thế ký tự lạ bằng _
        - Xóa phụ chú (in millions), (in thousands)
        - Thêm tiền tố col_ nếu bắt đầu bằng số
        - Đảm bảo tính duy nhất (deduplication)
        """
        if raw_name is None or str(raw_name).strip() == "":
            base_col = "item_name" if col_idx == 0 else f"col_{col_idx}"
        else:
            name_str = str(raw_name).strip()
            name_str = re.sub(r"\(in\s+millions?\)", "", name_str, flags=re.IGNORECASE)
            name_str = re.sub(r"\(in\s+thousands?\)", "", name_str, flags=re.IGNORECASE)
            name_str = re.sub(r"<[^>]+>", "", name_str)
            cleaned = re.sub(r"[^\w\s]", "_", name_str)
            cleaned = re.sub(r"\s+", "_", cleaned).strip("_").lower()

            if not cleaned or cleaned[0].isdigit():
                cleaned = f"col_{cleaned}"
            base_col = cleaned

        candidate = base_col
        counter = 1
        while candidate in existing_cols:
            candidate = f"{base_col}_{counter}"
            counter += 1

        existing_cols.add(candidate)
        return candidate

    def process_raw_table(
        self,
        raw_table: List[List[Any]],
        table_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Xử lý bảng 2D thô thành DataFrame chuẩn hóa cao cấp:
        - Xử lý hierarchical / merged headers (hàng 0 chứa niên độ).
        - Ép kiểu số REAL cho các cột định lượng.
        - Bổ sung `clean_item_name` và `is_total_row`.
        """
        if not raw_table or not isinstance(raw_table, list) or len(raw_table) < 2:
            return None

        max_cols = max(len(r) if isinstance(r, list) else 0 for r in raw_table)
        if max_cols == 0:
            return None

        header_raw = [str(c).strip() if c is not None else "" for c in raw_table[0]]
        if len(header_raw) < max_cols:
            header_raw += [f"col_{i}" for i in range(len(header_raw), max_cols)]

        data_start_row = 1
        if len(raw_table) >= 3:
            row_1 = [str(c).strip() if c is not None else "" for c in raw_table[1]]
            r1_nums = [ExtendedNumericSanitizer.parse_cell(c)[1] for c in row_1[1:] if c]
            r2_nums = [ExtendedNumericSanitizer.parse_cell(c)[1] for c in raw_table[2][1:] if c]

            if (len(r1_nums) == 0 or sum(r1_nums) == 0) and (len(r2_nums) > 0 and sum(r2_nums) / len(r2_nums) >= 0.5):
                merged_headers = []
                for idx in range(max_cols):
                    h0 = header_raw[idx] if idx < len(header_raw) else ""
                    h1 = row_1[idx] if idx < len(row_1) else ""
                    if h0 and h1 and h0.lower() != h1.lower():
                        merged_headers.append(f"{h0}_{h1}")
                    else:
                        merged_headers.append(h1 or h0 or f"col_{idx}")
                header_raw = merged_headers
                data_start_row = 2

        existing_cols = set()
        sanitized_cols = [
            self.sanitize_column_name(h, idx, existing_cols)
            for idx, h in enumerate(header_raw[:max_cols])
        ]

        data_rows = []
        for r_idx in range(data_start_row, len(raw_table)):
            row = raw_table[r_idx]
            if not isinstance(row, list):
                row = [row]
            if len(row) < max_cols:
                row = row + [""] * (max_cols - len(row))
            else:
                row = row[:max_cols]
            data_rows.append(row)

        if not data_rows:
            return None

        col_data = {col: [] for col in sanitized_cols}
        for col_idx, col_name in enumerate(sanitized_cols):
            raw_col_values = [r[col_idx] for r in data_rows]
            parsed_results = [ExtendedNumericSanitizer.parse_cell(v) for v in raw_col_values]

            numeric_hits = sum(1 for val, is_num in parsed_results if is_num and val is not None)
            total_non_empty = sum(1 for v in raw_col_values if str(v).strip() not in {"", "-", "—", "N/A"})

            if col_idx == 0:
                is_numeric_col = False
            elif total_non_empty > 0 and (numeric_hits / len(raw_col_values) >= 0.5):
                is_numeric_col = True
            elif total_non_empty == 0 and numeric_hits > 0 and col_idx > 0:
                is_numeric_col = True
            else:
                is_numeric_col = False

            if is_numeric_col:
                col_data[col_name] = [
                    float(parsed_val) if (is_num and parsed_val is not None) else 0.0
                    for parsed_val, is_num in parsed_results
                ]
            else:
                col_data[col_name] = [str(v).strip() if v is not None else "" for v in raw_col_values]

        df = pd.DataFrame(col_data)

        primary_text_col = sanitized_cols[0]
        df["clean_item_name"] = df[primary_text_col].apply(self.clean_text_label)
        df["is_total_row"] = df[primary_text_col].apply(self.check_is_total_row)

        return df

    def run_etl(self) -> Dict[str, Any]:
        """Thực thi toàn bộ luồng ETL trên toàn bộ 1,147 bảng."""
        logger.info("[ETL] Bắt đầu quy trình chuẩn hóa CSDL tài chính V2...")
        self.backup_existing_db()

        if not self.raw_json_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file nguồn: {self.raw_json_path}")

        with open(self.raw_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info(f"[ETL] Đã tải {len(raw_data)} mẫu từ {self.raw_json_path}.")

        temp_db_path = Path("data/database/finance_temp.db")
        if temp_db_path.exists():
            temp_db_path.unlink()

        conn = sqlite3.connect(str(temp_db_path))

        tables_processed = 0
        total_real_columns = 0
        total_total_rows = 0
        complex_table_logs = []

        for idx, item in enumerate(raw_data):
            table_id = f"table_{idx}"
            raw_table = item.get("table_ori") or item.get("table")
            if not raw_table:
                continue

            df = self.process_raw_table(raw_table, table_id=table_id)
            if df is None or df.empty:
                continue

            real_cols = [c for c in df.columns if df[c].dtype == "float64"]
            total_real_columns += len(real_cols)
            total_total_rows += int(df["is_total_row"].sum())

            df.to_sql(table_id, con=conn, if_exists="replace", index=True, index_label="row_id")
            tables_processed += 1

            if idx in [0, 1, 2]:
                complex_table_logs.append({
                    "table_id": table_id,
                    "columns": list(df.columns),
                    "real_columns": real_cols,
                    "total_rows_marked": int(df["is_total_row"].sum()),
                    "sample_records": df.head(3).to_dict(orient="records")
                })

            if tables_processed % 100 == 0 or tables_processed == len(raw_data):
                logger.info(f"[ETL] Đã xử lý & chuẩn hóa {tables_processed}/{len(raw_data)} bảng...")

        conn.execute("VACUUM;")
        conn.close()

        if self.db_path.exists():
            self.db_path.unlink()
        temp_db_path.rename(self.db_path)
        logger.success(f"[ETL] Đã thay thế thành công CSDL chuẩn hóa V2 tại: {self.db_path}")

        summary = {
            "tables_processed": tables_processed,
            "total_real_columns": total_real_columns,
            "total_total_rows_marked": total_total_rows,
            "complex_table_logs": complex_table_logs,
        }

        with open("docs/etl_v2_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    etl = FinancialDataETL()
    summary = etl.run_etl()
    print("\n" + "=" * 70)
    print("ETL V2 SUMMARY REPORT:")
    print(f"- Tables processed:    {summary['tables_processed']}")
    print(f"- Columns cast to REAL: {summary['total_real_columns']}")
    print(f"- Rows marked is_total: {summary['total_total_rows_marked']}")
    print("=" * 70)
