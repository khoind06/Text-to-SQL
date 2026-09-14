"""
scripts/download_spider.py
Tự động tải và chuẩn bị tập dữ liệu Spider (Yale Text-to-SQL Benchmark).

Chức năng:
1. Tải tập train và validation từ HuggingFace (datasets.load_dataset("spider") hoặc xlangai/spider).
2. Tải và giải nén toàn bộ các file CSDL SQLite vật lý vào data/spider/database/{db_id}/{db_id}.sqlite.
3. Trích xuất nhãn vàng (gold SQL queries) của tập validation vào data/spider/dev_gold.json.
"""

import json
import os
import re
import sys
import zipfile
from pathlib import Path

# Cấu hình UTF-8 cho Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import datasets
from loguru import logger

SPIDER_ZIP_URL = "https://drive.google.com/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
DATA_DIR = Path("data/spider")
DATABASE_DIR = DATA_DIR / "database"
DEV_GOLD_PATH = DATA_DIR / "dev_gold.json"
ZIP_PATH = Path("data/spider.zip")


def load_spider_hf():
    """
    Tải tập dữ liệu Spider qua thư viện datasets của HuggingFace.
    Tự động fallback sang xlangai/spider nếu spider gốc bị lỗi định dạng trên HF.
    """
    logger.info("Đang nạp tập dữ liệu Spider từ HuggingFace...")
    try:
        ds = datasets.load_dataset("spider")
        logger.info("Tải thành công dataset 'spider' từ HuggingFace.")
        return ds
    except Exception as e:
        logger.warning(f"Không thể nạp trực tiếp 'spider' ({e}). Thử nạp mirror 'xlangai/spider'...")
        ds = datasets.load_dataset("xlangai/spider")
        logger.info("Tải thành công dataset 'xlangai/spider' từ HuggingFace.")
        return ds


def extract_databases(zip_path: Path, target_dir: Path):
    """
    Giải nén các file SQLite vật lý từ file zip của Spider vào thư mục target_dir.
    Đảm bảo cấu trúc: target_dir/{db_id}/{db_id}.sqlite
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if not zip_path.exists():
        logger.info(f"File {zip_path} chưa tồn tại. Đang tải từ Google Drive...")
        import gdown
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(SPIDER_ZIP_URL, str(zip_path), quiet=False)

    logger.info(f"Đang giải nén các CSDL SQLite từ {zip_path} vào {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Bỏ qua các file rác MacOS
            if "__MACOSX" in member.filename:
                continue

            # Tìm các file nằm trong thư mục database/
            # Cấu trúc trong zip có thể là spider_data/database/... hoặc database/...
            filename = member.filename.replace("\\", "/")
            if "database/" in filename:
                idx = filename.find("database/") + len("database/")
                sub_path = filename[idx:]
                if not sub_path or sub_path.endswith("/"):
                    continue

                # Làm sạch tên file không hợp lệ trên Windows (ví dụ dấu hai chấm :)
                sanitized_sub_path = re.sub(r'[:*?"<>|]', '_', sub_path)
                dest_file = target_dir / sanitized_sub_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    with z.open(member) as source, open(dest_file, "wb") as target:
                        target.write(source.read())
                except Exception as e:
                    logger.warning(f"Bỏ qua file không tương thích: {member.filename} ({e})")

    # Đếm số lượng sqlite databases
    sqlite_files = list(target_dir.glob("*/*.sqlite"))
    logger.info(f"Giải nén thành công {len(sqlite_files)} CSDL SQLite vật lý tại: {target_dir}")
    return len(sqlite_files)


def export_dev_gold(hf_dataset, output_path: Path):
    """
    Trích xuất nhãn vàng từ tập validation của Spider thành file JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_data = hf_dataset["validation"]
    
    gold_records = []
    for idx, item in enumerate(val_data):
        gold_records.append({
            "index": idx,
            "db_id": item["db_id"],
            "question": item["question"],
            "query": item["query"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_records, f, ensure_ascii=False, indent=2)

    logger.info(f"Đã trích xuất {len(gold_records)} mẫu nhãn vàng vào: {output_path}")
    return len(gold_records)


def main():
    logger.info("=== BẮT ĐẦU CHUẨN BỊ DỮ LIỆU SPIDER ===")
    
    # 1. Nạp từ HuggingFace
    hf_ds = load_spider_hf()
    train_count = len(hf_ds["train"]) if "train" in hf_ds else 0
    val_count = len(hf_ds["validation"]) if "validation" in hf_ds else 0
    logger.info(f"HuggingFace Splits: Train = {train_count} mẫu | Validation = {val_count} mẫu")

    # 2. Giải nén kho SQLite vật lý
    num_dbs = extract_databases(ZIP_PATH, DATABASE_DIR)

    # 3. Trích xuất file dev_gold.json
    num_gold = export_dev_gold(hf_ds, DEV_GOLD_PATH)

    logger.info("=== HOÀN TẤT CHUẨN BỊ DỮ LIỆU SPIDER ===")
    logger.info(f"✓ CSDL vật lý: {num_dbs} databases tại {DATABASE_DIR}")
    logger.info(f"✓ Nhãn vàng dev: {num_gold} mẫu tại {DEV_GOLD_PATH}")


if __name__ == "__main__":
    main()
