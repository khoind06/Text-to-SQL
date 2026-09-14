"""
scripts/evaluate_spider.py
Benchmark Text-to-SQL thuần túy trên tập dữ liệu Spider (Yale Benchmark).

Đánh giá chuẩn Execution Accuracy (EX):
- So sánh tập kết quả trả về của SQL do mô hình sinh ra và SQL nhãn vàng trên cùng CSDL SQLite.
- Bỏ qua tên cột (alias) và thứ tự dòng nếu không có mệnh đề ORDER BY.
- Phân loại lỗi chi tiết: SYNTAX_ERROR, SCHEMA_ERROR, LOGIC_ERROR, GUARDRAIL_BLOCKED.
- Checkpointing lũy tiến lưu kết quả mỗi 10 câu vào docs/spider_evaluation_results.json.
- Thanh tiến trình trực quan qua tqdm, tối ưu hóa I/O console.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm import tqdm

# Cấu hình UTF-8 cho Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo import được các module từ src/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger
from src.agents.orchestrator import FinancialOrchestrator


def normalize_val(val: Any) -> Any:
    """Chuẩn hóa giá trị để so sánh độc lập với kiểu dữ liệu thô."""
    if val is None:
        return None
    if isinstance(val, (int, bool)):
        return float(val)
    if isinstance(val, float):
        return round(val, 4)
    val_str = str(val).strip()
    try:
        f = float(val_str)
        return round(f, 4)
    except ValueError:
        return val_str.lower()


def execute_sql_raw(db_path: str, sql: str) -> Tuple[Optional[List[Tuple]], Optional[str]]:
    """Thực thi câu SQL trực tiếp bằng sqlite3 để lấy kết quả thô không phụ thuộc tên cột."""
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def compare_results(gold_rows: List[Tuple], pred_rows: List[Tuple], has_order_by: bool) -> bool:
    """
    So sánh hai tập kết quả theo chuẩn Spider Execution Accuracy (EX).
    - Nếu has_order_by=True: So sánh tuần tự từng dòng.
    - Nếu has_order_by=False: So sánh theo đa tập hợp (Multiset) không phụ thuộc thứ tự dòng.
    """
    if len(gold_rows) != len(pred_rows):
        return False
    if len(gold_rows) == 0 and len(pred_rows) == 0:
        return True

    norm_gold = [tuple(normalize_val(v) for v in row) for row in gold_rows]
    norm_pred = [tuple(normalize_val(v) for v in row) for row in pred_rows]

    if has_order_by:
        return norm_gold == norm_pred

    try:
        from collections import Counter
        return Counter(norm_gold) == Counter(norm_pred)
    except Exception:
        return sorted([str(r) for r in norm_gold]) == sorted([str(r) for r in norm_pred])


def categorize_error(error_msg: str, pred_sql: str) -> str:
    """Phân loại nguyên nhân lỗi truy vấn."""
    if not error_msg:
        return "LOGIC_ERROR"
    err_lower = error_msg.lower()
    if "security" in err_lower or "bảo mật" in err_lower or "ast" in err_lower:
        return "GUARDRAIL_BLOCKED"
    if "no such table" in err_lower or "no such column" in err_lower or "has no column" in err_lower:
        return "SCHEMA_ERROR"
    if "syntax error" in err_lower or "incomplete input" in err_lower or "operationalerror" in err_lower:
        return "SYNTAX_ERROR"
    return "EXECUTION_ERROR"


def save_checkpoint(checkpoint_file: str, checkpoint_data: Dict[str, Any]):
    """Ghi dữ liệu checkpoint lũy tiến."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    temp_file = f"{checkpoint_file}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
        except Exception:
            pass
    os.rename(temp_file, checkpoint_file)


def run_evaluation(
    samples_limit: Optional[int] = None,
    offset: int = 0,
    model_name: str = "qwen2.5-coder:7b",
    gold_path: str = "data/spider/dev_gold.json",
    db_base_dir: str = "data/spider/database",
    checkpoint_file: str = "docs/spider_evaluation_results.json",
    fresh: bool = False,
    quiet: bool = True
):
    """Quy trình đánh giá benchmark trên tập Spider validation."""
    # Tắt log console thừa thãi, ghi toàn bộ log chi tiết vào file logs/spider_benchmark.log
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add("logs/spider_benchmark.log", rotation="50 MB", level="INFO", encoding="utf-8")

    print("\n" + "=" * 80)
    print("🚀 KHỞI ĐỘNG BENCHMARK TEXT-TO-SQL TRÊN TOÀN BỘ TẬP SPIDER VALIDATION")
    print(f"   Model: {model_name} | Nhãn vàng: {gold_path}")
    print(f"   Kho CSDL: {db_base_dir} | Checkpoint: {checkpoint_file}")
    print(f"   Chế độ: {'Chạy mới từ đầu (Fresh)' if fresh else 'Tiếp tục từ checkpoint (Resume)'}")
    print("=" * 80 + "\n")

    # 1. Nạp danh sách câu hỏi nhãn vàng
    if not os.path.exists(gold_path):
        print(f"❌ LỖI: Không tìm thấy file nhãn vàng tại {gold_path}.")
        sys.exit(1)

    with open(gold_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    selected_samples = all_samples[offset:]
    if samples_limit is not None:
        selected_samples = selected_samples[:samples_limit]

    total_samples = len(selected_samples)
    print(f"📋 Tổng số mẫu thực thi đợt này: {total_samples} / {len(all_samples)} mẫu\n")

    # 2. Xử lý Checkpoint lũy tiến
    completed_results: List[Dict[str, Any]] = []
    completed_indices: Set[int] = set()
    if not fresh and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                completed_results = saved_data.get("results", [])
                completed_indices = {r["index"] for r in completed_results}
                print(f"🔄 Đã phục hồi Checkpoint: {len(completed_indices)} mẫu đã hoàn thành trước đó.")
        except Exception as e:
            print(f"⚠️ Không thể đọc Checkpoint ({e}). Sẽ chạy mới.")

    # 3. Khởi tạo Orchestrator Single-Agent
    orchestrator = FinancialOrchestrator(sql_model=model_name)

    passed_count = sum(1 for r in completed_results if r.get("status") == "PASS")
    total_latency_accum = sum(r.get("latency_seconds", 0.0) for r in completed_results)
    start_bench_time = time.perf_counter()

    pbar = tqdm(selected_samples, desc="Spider Benchmark (EX)", unit="mẫu", dynamic_ncols=True)

    for idx, sample in enumerate(pbar, 1):
        sample_idx = sample["index"]
        if sample_idx in completed_indices:
            pbar.set_postfix({
                "PASS": passed_count,
                "EX": f"{(passed_count / len(completed_indices) * 100):.1f}%" if completed_indices else "0.0%",
            })
            continue

        q = sample["question"]
        gold_sql = sample["query"]
        db_id = sample["db_id"]
        db_path = os.path.join(db_base_dir, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            tqdm.write(f"❌ [Index {sample_idx:04d}] Thiếu CSDL: {db_path}")
            continue

        has_order_by = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))
        t0 = time.perf_counter()

        # Thực thi Single-Agent SQL
        state = orchestrator.run(question=q, db_id=db_id, db_path=db_path)
        latency = round(time.perf_counter() - t0, 3)
        total_latency_accum += latency

        pred_sql = state.get("sql_query") or ""
        agent_error = state.get("error")

        # Thực thi SQL thực tế trên SQLite
        pred_rows, pred_err = execute_sql_raw(db_path, pred_sql) if pred_sql else (None, "Empty SQL")
        gold_rows, gold_err = execute_sql_raw(db_path, gold_sql)

        # Đánh giá Execution Accuracy
        status = "FAIL"
        failure_category = None

        if pred_err is not None:
            failure_category = categorize_error(pred_err, pred_sql)
        elif gold_err is not None:
            failure_category = "GOLD_SQL_ERROR"
        else:
            is_match = compare_results(gold_rows, pred_rows, has_order_by)
            if is_match:
                status = "PASS"
                passed_count += 1
            else:
                failure_category = "LOGIC_ERROR"

        # Cập nhật thông tin tóm tắt lỗi ra console
        if status == "FAIL":
            tqdm.write(
                f"[{sample_idx:04d}] ❌ FAIL [{failure_category}] | DB: {db_id} | Q: {q[:60]}... | "
                f"Pred: {pred_sql[:40]}..."
            )

        record = {
            "index": sample_idx,
            "db_id": db_id,
            "question": q,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "status": status,
            "failure_category": failure_category,
            "pred_error": pred_err,
            "gold_error": gold_err,
            "gold_row_count": len(gold_rows) if gold_rows is not None else 0,
            "pred_row_count": len(pred_rows) if pred_rows is not None else 0,
            "latency_seconds": latency,
            "retries": state.get("retry_count", 0),
        }
        completed_results.append(record)
        completed_indices.add(sample_idx)

        # Cập nhật thanh tiến trình
        evaluated_so_far = len(completed_results)
        ex_current = round((passed_count / evaluated_so_far) * 100, 1)
        pbar.set_postfix({
            "PASS": f"{passed_count}/{evaluated_so_far}",
            "EX": f"{ex_current}%",
            "Lat": f"{latency:.2f}s"
        })

        # Checkpointing lũy tiến sau mỗi 10 câu hoặc câu cuối
        if len(completed_results) % 10 == 0 or idx == total_samples:
            checkpoint_data = {
                "benchmark_name": "Spider_Validation_Single_Agent",
                "model_name": model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_evaluated": evaluated_so_far,
                "passed_count": passed_count,
                "execution_accuracy_pct": round((passed_count / evaluated_so_far) * 100, 2),
                "average_latency_seconds": round(total_latency_accum / evaluated_so_far, 3),
                "results": completed_results
            }
            save_checkpoint(checkpoint_file, checkpoint_data)

    pbar.close()

    total_bench_time = round(time.perf_counter() - start_bench_time, 2)
    total_evaluated = len(completed_results)
    final_accuracy = round((passed_count / total_evaluated) * 100, 2) if total_evaluated > 0 else 0.0
    avg_latency = round(total_latency_accum / total_evaluated, 3) if total_evaluated > 0 else 0.0

    # Phân tích nguyên nhân lỗi
    from collections import Counter
    failures = [r for r in completed_results if r["status"] == "FAIL"]
    failure_counts = Counter(r["failure_category"] for r in failures)
    total_fails = len(failures)

    print("\n" + "=" * 80)
    print("🏆 BÁO CÁO TỔNG KẾT BENCHMARK SPIDER VALIDATION (1,034 MẪU)")
    print("=" * 80)
    print(f"📊 Tổng số mẫu kiểm thử:           {total_evaluated} / 1,034")
    print(f"🎯 Execution Accuracy (EX):         {final_accuracy:.2f}% ({passed_count}/{total_evaluated})")
    print(f"⚡ Trung bình độ trễ (Avg Latency): {avg_latency:.3f} giây / câu")
    print(f"⏱️ Tổng thời gian chạy:            {total_bench_time:.2f} giây ({total_bench_time / 60:.1f} phút)")
    print("-" * 80)
    print(f"🔍 PHÂN LOẠI LỖI (ERROR BREAKDOWN - TỔNG {total_fails} CÂU LỖI):")
    for cat, cnt in failure_counts.most_common():
        pct = (cnt / total_fails * 100) if total_fails > 0 else 0.0
        pct_of_all = (cnt / total_evaluated * 100) if total_evaluated > 0 else 0.0
        print(f"  • {cat:20s}: {cnt:4d} câu ({pct:5.1f}% số lỗi | {pct_of_all:5.1f}% toàn tập)")
    print("=" * 80 + "\n")

    return checkpoint_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Benchmark Spider Text-to-SQL")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ 1,034 mẫu")
    parser.add_argument("--samples", type=int, default=None, help="Số lượng mẫu kiểm thử (mặc định toàn bộ)")
    parser.add_argument("--offset", type=int, default=0, help="Vị trí bắt đầu trong tập validation")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:7b", help="Tên mô hình LLM")
    parser.add_argument("--checkpoint", type=str, default="docs/spider_evaluation_results.json", help="File checkpoint JSON")
    parser.add_argument("--fresh", action="store_true", help="Xóa checkpoint và chạy mới từ đầu")
    args = parser.parse_args()

    limit = None if args.all else args.samples

    run_evaluation(
        samples_limit=limit,
        offset=args.offset,
        model_name=args.model,
        checkpoint_file=args.checkpoint,
        fresh=args.fresh
    )
