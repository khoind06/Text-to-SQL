"""
scripts/isolate_ground_truth.py
Script thực hiện cách ly hoàn toàn dữ liệu ground truth (finqa_metadata) ra khỏi finance.db.
Tuân thủ tiêu chuẩn Zero-Trust:
1. Trích xuất toàn bộ dữ liệu từ bảng finqa_metadata sang data/evaluation_ground_truth.json.
2. DROP TABLE finqa_metadata khỏi CSDL finance.db.
3. Thực thi VACUUM để tối ưu hóa và dọn dẹp vật lý CSDL.
"""

import json
import sqlite3
from pathlib import Path
from loguru import logger


def isolate_ground_truth(
    db_path: str = "data/database/finance.db",
    output_json: str = "data/evaluation_ground_truth.json"
):
    db_file = Path(db_path)
    out_file = Path(output_json)

    if not db_file.exists():
        logger.error(f"[Isolation] Không tìm thấy cơ sở dữ liệu tại: {db_file}")
        raise FileNotFoundError(f"Database file not found: {db_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Isolation] Đang kết nối trực tiếp tới SQLite DB: {db_file}")
    conn = sqlite3.connect(str(db_file.resolve()))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # 1. Kiểm tra sự tồn tại của bảng finqa_metadata
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='finqa_metadata';")
        table_exists = cursor.fetchone()

        if table_exists:
            logger.info("[Isolation] Phát hiện bảng 'finqa_metadata'. Đang trích xuất dữ liệu ground truth...")
            cursor.execute("SELECT * FROM finqa_metadata;")
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            logger.info(f"[Isolation] Đã trích xuất thành công {len(data)} bản ghi từ 'finqa_metadata'.")

            # Lưu vào file JSON độc lập
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.success(f"[Isolation] Đã lưu Ground Truth ra file độc lập: {out_file.resolve()}")

            # 2. DROP TABLE finqa_metadata
            logger.warning("[Isolation] Đang thực thi: DROP TABLE finqa_metadata;")
            cursor.execute("DROP TABLE finqa_metadata;")
            conn.commit()
            logger.success("[Isolation] Bảng 'finqa_metadata' đã bị XÓA VĨNH VIỄN khỏi finance.db.")
        else:
            logger.warning("[Isolation] Bảng 'finqa_metadata' không tồn tại trong database (có thể đã được cách ly trước đó).")

        # 3. VACUUM CSDL
        logger.info("[Isolation] Đang thực thi VACUUM để thu hồi bộ nhớ vật lý...")
        conn.close()

        # VACUUM cần kết nối ở chế độ autocommit (isolation_level=None)
        vac_conn = sqlite3.connect(str(db_file.resolve()), isolation_level=None)
        vac_conn.execute("VACUUM;")
        vac_conn.close()
        logger.success("[Isolation] VACUUM hoàn tất thành công. finance.db đã đạt chuẩn Zero-Trust!")

    except Exception as e:
        logger.error(f"[Isolation] Thất bại trong quá trình cách ly dữ liệu: {e}")
        raise


if __name__ == "__main__":
    isolate_ground_truth()
