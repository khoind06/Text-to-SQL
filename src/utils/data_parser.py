"""
data_parser.py
Utility module to parse FinQA dataset (JSON format) and ingest financial tables
into a relational SQLite database with robust schema sanitization, uneven row padding,
and metadata indexing.
"""

import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from loguru import logger


def sanitize_column_name(name: Any, col_idx: int, existing_names: set) -> str:
    """
    Sanitize and ensure uniqueness of column names for SQL compliance.
    Replaces special characters with underscores, ensures valid SQL identifiers.
    """
    if name is None or str(name).strip() == "":
        col_name = "item_name" if col_idx == 0 else f"col_{col_idx}"
    else:
        # Convert to string and clean special characters
        cleaned = re.sub(r"[^\w\s]", "_", str(name).strip())
        cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
        
        # If starts with a digit or is empty, prefix with col_
        if not cleaned or cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        col_name = cleaned.lower()

    # Ensure uniqueness
    base_name = col_name
    counter = 1
    while col_name in existing_names:
        col_name = f"{base_name}_{counter}"
        counter += 1

    existing_names.add(col_name)
    return col_name


def parse_raw_table_to_df(raw_table: List[List[Any]], sample_id: str = "") -> Optional[pd.DataFrame]:
    """
    Convert a 2D list of table data from FinQA into a clean pandas DataFrame.
    Handles:
    - Empty or single-row tables
    - Uneven row lengths across rows
    - Column name deduplication and SQL identifier sanitization
    """
    if not raw_table or not isinstance(raw_table, list) or len(raw_table) < 2:
        logger.warning(f"[{sample_id}] Table has fewer than 2 rows. Skipping.")
        return None

    # Calculate maximum columns across all rows to handle uneven rows
    max_cols = max(len(row) if isinstance(row, list) else 0 for row in raw_table)
    if max_cols == 0:
        logger.warning(f"[{sample_id}] All rows are empty. Skipping.")
        return None

    # Pad header row if needed
    header_raw = raw_table[0] if isinstance(raw_table[0], list) else []
    if len(header_raw) < max_cols:
        header_raw = header_raw + [f"extra_col_{i}" for i in range(len(header_raw), max_cols)]

    # Sanitize header names
    existing_cols = set()
    sanitized_headers = [sanitize_column_name(col, idx, existing_cols) for idx, col in enumerate(header_raw[:max_cols])]

    # Normalize data rows
    data_rows = []
    uneven_row_count = 0

    for r_idx, row in enumerate(raw_table[1:], start=1):
        if not isinstance(row, list):
            row = [str(row)]
        
        if len(row) < max_cols:
            uneven_row_count += 1
            # Pad with empty string
            padded_row = row + [""] * (max_cols - len(row))
        elif len(row) > max_cols:
            uneven_row_count += 1
            padded_row = row[:max_cols]
        else:
            padded_row = row

        # Clean string representation
        cleaned_cells = ["" if cell is None else str(cell).strip() for cell in padded_row]
        data_rows.append(cleaned_cells)

    if uneven_row_count > 0:
        logger.debug(f"[{sample_id}] Padded {uneven_row_count} uneven rows to {max_cols} columns.")

    df = pd.DataFrame(data_rows, columns=sanitized_headers)
    return df


def parse_finqa_to_sqlite(
    json_path: str = "FinQA/dataset/train.json",
    db_path: str = "data/database/finance.db",
    max_tables: int = 100,
    use_table_ori: bool = True,
    if_exists: str = "replace"
) -> Dict[str, Any]:
    """
    Parse FinQA JSON dataset and export tables + metadata to SQLite database.

    Args:
        json_path: Path to FinQA train.json (or dev/test.json).
        db_path: Destination path for SQLite database.
        max_tables: Number of tables to process (default 100).
        use_table_ori: Prefer 'table_ori' (original capitalization & punctuation) if available.
        if_exists: How to behave if metadata/tables exist ('replace' or 'append').

    Returns:
        Summary dict containing statistics of processed tables.
    """
    json_file = Path(json_path)
    if not json_file.exists():
        alt_path = Path("FinQA/train.json")
        if alt_path.exists():
            json_file = alt_path
        else:
            raise FileNotFoundError(f"Cannot find FinQA dataset file at {json_path} or {alt_path}")

    # Ensure database directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{db_file.resolve().as_posix()}"
    engine = create_engine(db_uri, echo=False)

    logger.info(f"Loading FinQA dataset from: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_samples = len(dataset)
    process_limit = min(max_tables, total_samples) if max_tables > 0 else total_samples
    logger.info(f"Loaded {total_samples} samples. Processing up to {process_limit} tables...")

    metadata_records = []
    successful_tables = 0
    failed_tables = 0

    for idx in range(process_limit):
        item = dataset[idx]
        sample_id = item.get("id", f"sample_{idx}")
        filename = item.get("filename", "")
        
        # QA information
        qa_data = item.get("qa", {})
        question = qa_data.get("question") or item.get("question", "")
        program = qa_data.get("program", "")
        exe_ans = str(qa_data.get("exe_ans", ""))
        gold_inds = json.dumps(qa_data.get("gold_inds", {}), ensure_ascii=False)
        explanation = qa_data.get("explanation", "")
        
        pre_text = json.dumps(item.get("pre_text", []), ensure_ascii=False)
        post_text = json.dumps(item.get("post_text", []), ensure_ascii=False)

        # Select table source
        raw_table = None
        if use_table_ori and "table_ori" in item and item["table_ori"]:
            raw_table = item["table_ori"]
        elif "table" in item and item["table"]:
            raw_table = item["table"]

        if not raw_table:
            logger.warning(f"[{sample_id}] No table found in entry {idx}. Skipping.")
            failed_tables += 1
            continue

        try:
            df = parse_raw_table_to_df(raw_table, sample_id=sample_id)
            if df is None or df.empty:
                failed_tables += 1
                continue

            table_name = f"table_{successful_tables}"
            
            # Write individual table to SQLite
            df.to_sql(table_name, con=engine, if_exists=if_exists, index=True, index_label="row_id")

            # Collect metadata
            metadata_records.append({
                "table_id": table_name,
                "sample_id": sample_id,
                "filename": filename,
                "question": question,
                "program": program,
                "exe_ans": exe_ans,
                "gold_inds": gold_inds,
                "explanation": explanation,
                "pre_text": pre_text,
                "post_text": post_text,
                "num_rows": len(df),
                "num_cols": len(df.columns),
                "columns_json": json.dumps(list(df.columns), ensure_ascii=False)
            })

            successful_tables += 1
            if successful_tables % 20 == 0 or successful_tables == process_limit:
                logger.info(f"Ingested {successful_tables}/{process_limit} tables into SQLite...")

        except Exception as e:
            logger.error(f"[{sample_id}] Error converting table {idx} to SQL: {str(e)}")
            failed_tables += 1

    # Save metadata table
    if metadata_records:
        meta_df = pd.DataFrame(metadata_records)
        meta_df.to_sql("finqa_metadata", con=engine, if_exists=if_exists, index=False)
        logger.success(f"Saved 'finqa_metadata' table with {len(meta_df)} entries.")

    summary = {
        "database_path": str(db_file.resolve()),
        "tables_created": successful_tables,
        "tables_failed": failed_tables,
        "total_attempted": process_limit,
        "metadata_table": "finqa_metadata"
    }

    logger.success(
        f"FinQA SQLite preparation complete: {successful_tables} tables created, "
        f"{failed_tables} failed. DB Location: {db_file}"
    )
    return summary


if __name__ == "__main__":
    summary = parse_finqa_to_sqlite(
        json_path="FinQA/dataset/train.json",
        db_path="data/database/finance.db",
        max_tables=100
    )
    print("\nSummary Result:")
    print(json.dumps(summary, indent=2))
