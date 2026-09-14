"""
run_two_stage_benchmark.py
Automated 100-Sample Benchmark for Two-Stage Execution (SQL Retrieval + Python Quant Engine).
Features:
- quant_only=True: bypasses Agent 3 (CRO) for maximum speed and focus on numerical reasoning.
- 100 random samples with reproducible seed (random.seed(42)) from data/evaluation_ground_truth.json.
- Progressive checkpointing into docs/two_stage_100_pilot.json.
- Full metric aggregation: Execution Accuracy (Exact Match <= 1%), SQL Extraction Success Rate,
  Python Execution Success Rate, Latency, and Top 3 Failure Modes analysis.
"""

import sys
import re
import math
import time
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.orchestrator import FinancialOrchestrator


def evaluate_quant_accuracy(quant_val: Any, gold_answer: str) -> Tuple[int, str]:
    """
    So khớp định lượng nghiêm ngặt giữa kết quả Python Sandbox và đáp án vàng exe_ans:
    1. Boolean matching: "yes" / "no".
    2. Numerical matching: math.isclose với rel_tol=0.01 (sai số <= 1%) và abs_tol=1e-4.
    3. Percentage scaling: chấp nhận sai lệch hệ số 100 (ví dụ 0.1446 so với 14.46).
    """
    if quant_val is None:
        return 0, "FAIL: Không có kết quả tính toán định lượng (quant_result is None)."

    gold_clean = str(gold_answer).strip()
    gold_clean_lower = gold_clean.lower()

    # 1. Khớp nhãn Boolean (Yes / No)
    if gold_clean_lower in ["yes", "no"]:
        str_val = str(quant_val).strip().lower()
        if str_val == gold_clean_lower or (gold_clean_lower == "yes" and quant_val is True) or (gold_clean_lower == "no" and quant_val is False):
            return 1, f"Khớp nhãn Boolean '{gold_clean}'."
        return 0, f"FAIL: Sai lệch nhãn Boolean. Kỳ vọng '{gold_clean}', thực tế: '{quant_val}'."

    # 2. Parse số vàng (gold_val)
    gold_cleaned_num_str = re.sub(r"[^\d\.-]", "", gold_clean)
    try:
        gold_num = float(gold_cleaned_num_str)
    except ValueError:
        if str(quant_val).strip().lower() == gold_clean_lower:
            return 1, f"Khớp chuỗi chính xác '{gold_clean}'."
        return 0, f"FAIL: Không thể parse số vàng '{gold_clean}'."

    # 3. Parse số do Python Sandbox nhả ra
    try:
        if isinstance(quant_val, (int, float)):
            pred_num = float(quant_val)
        elif isinstance(quant_val, str):
            clean_str = re.sub(r"[^\d\.-]", "", quant_val.strip())
            pred_num = float(clean_str)
        elif hasattr(quant_val, "iloc"):
            pred_num = float(quant_val.iloc[0])
        elif isinstance(quant_val, (list, tuple)) and len(quant_val) > 0:
            pred_num = float(quant_val[0])
        else:
            pred_num = float(quant_val)
    except Exception as e:
        return 0, f"FAIL: Lỗi ép kiểu số: {e} ({quant_val})"

    if math.isnan(pred_num) or math.isinf(pred_num):
        return 0, "FAIL: Kết quả tính toán là NaN hoặc Infinity."

    # 4. So sánh số học chính xác (rel_tol <= 1%)
    if math.isclose(pred_num, gold_num, rel_tol=0.01, abs_tol=1e-4):
        return 1, f"Khớp chính xác: {pred_num} == Vàng {gold_num} (sai số <= 1%)."

    # Khớp tỷ lệ phần trăm (nhân 100 hoặc chia 100)
    if math.isclose(pred_num * 100, gold_num, rel_tol=0.01, abs_tol=1e-4):
        return 1, f"Khớp phần trăm: {pred_num} * 100 == Vàng {gold_num}."
    if math.isclose(pred_num / 100, gold_num, rel_tol=0.01, abs_tol=1e-4):
        return 1, f"Khớp phần trăm: {pred_num} / 100 == Vàng {gold_num}."

    return 0, f"FAIL: Lệch số: Máy tính ra {pred_num}, Đáp án vàng {gold_num} (lệch {abs(pred_num - gold_num):.4f})."


def run_benchmark(
    sample_size: int = 100,
    checkpoint_path: str = "docs/two_stage_100_pilot.json",
    random_seed: int = 42,
    fresh: bool = False
):
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Ground Truth
    gt_file = Path("data/evaluation_ground_truth.json")
    if not gt_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {gt_file}")

    with open(gt_file, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    valid_records = [r for r in all_records if r.get("exe_ans") and str(r.get("exe_ans")).strip()]
    logger.info(f"Loaded {len(valid_records)} valid records from ground truth.")

    # 2. Lấy mẫu ngẫu nhiên 100 câu có seed cố định để tái hiện
    random.seed(random_seed)
    sampled_indices = random.sample(range(len(valid_records)), sample_size)
    sampled_indices.sort()
    test_samples = [valid_records[i] for i in sampled_indices]

    # 3. Khôi phục từ checkpoint nếu có
    results = []
    completed_keys = set()
    if not fresh and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict) and "results" in saved:
                    results = saved["results"]
                elif isinstance(saved, list):
                    results = saved
                for r in results:
                    completed_keys.add(r.get("sample_id") or r.get("index"))
            logger.success(f"[Checkpoint] Khôi phục {len(results)} câu đã chạy từ {checkpoint_file}")
        except Exception as e:
            logger.warning(f"[Checkpoint] Không thể đọc checkpoint: {e}")

    # 4. Khởi tạo Orchestrator
    logger.info("[Benchmark] Khởi tạo FinancialOrchestrator với qwen2.5-coder:7b (quant_only=True)...")
    orchestrator = FinancialOrchestrator(
        db_path="data/database/finance.db",
        sql_model="qwen2.5-coder:7b",
        quant_model="qwen2.5-coder:7b",
    )

    print("\n" + "="*80)
    print(f"🚀 KÍCH HOẠT BENCHMARK TWO-STAGE QUANTITATIVE ({sample_size} MẪU)")
    print(f"   Mode: quant_only=True (Bỏ qua Agent 3 CRO, chỉ lấy kết quả Sandbox)")
    print(f"   Checkpoint File: {checkpoint_file}")
    print(f"   Đã hoàn thành trước đó: {len(completed_keys)}/{sample_size}")
    print("="*80 + "\n")

    start_bench_time = time.time()

    for idx, sample in enumerate(test_samples, start=1):
        sample_key = sample.get("sample_id") or idx
        if sample_key in completed_keys:
            continue

        question = sample["question"]
        gold_ans = sample["exe_ans"]
        table_id = sample.get("table_id", "")

        # Inject table_id vào question nếu chưa có
        eval_question = f"In {table_id}: {question}" if table_id and table_id.lower() not in question.lower() else question

        t0 = time.perf_counter()
        try:
            state = orchestrator.run(eval_question, quant_only=True)
            latency = round(time.perf_counter() - t0, 3)
        except Exception as e:
            latency = round(time.perf_counter() - t0, 3)
            state = {
                "sql_query": "",
                "sql_result": None,
                "python_code": "",
                "quant_result": None,
                "error": f"CrashError: {str(e)}",
                "latency_trace": {}
            }

        sql_res = state.get("sql_result")
        sql_success = (sql_res is not None and isinstance(sql_res, pd.DataFrame) and len(sql_res) > 0)

        python_code = state.get("python_code")
        quant_res = state.get("quant_result")
        python_success = (python_code is not None and len(str(python_code).strip()) > 0 and quant_res is not None)

        # Đánh giá độ chính xác số học
        score, reason = evaluate_quant_accuracy(quant_res, gold_ans)

        # Phân loại nguyên nhân lỗi nếu FAIL
        failure_category = None
        if score == 0:
            if not sql_success:
                failure_category = "SQL_EMPTY_OR_FAIL"
            elif not python_success:
                failure_category = "PYTHON_CRASH_OR_SYNTAX"
            else:
                failure_category = "CALCULATION_MISMATCH"

        status_tag = "✅ PASS" if score == 1 else f"❌ FAIL [{failure_category}]"
        logger.info(
            f"[{idx:03d}/{sample_size}] {status_tag} | {latency:.2f}s | "
            f"Pred: {quant_res} | Gold: {gold_ans} | Table: {table_id}"
        )

        record = {
            "index": idx,
            "sample_id": sample.get("sample_id", f"sample_{idx}"),
            "table_id": table_id,
            "question": question,
            "gold_answer": gold_ans,
            "sql_query": state.get("sql_query", ""),
            "sql_success": sql_success,
            "sql_rows": len(sql_res) if isinstance(sql_res, pd.DataFrame) else 0,
            "python_code": python_code,
            "python_success": python_success,
            "quant_result": str(quant_res) if quant_res is not None else None,
            "score": score,
            "reason": reason,
            "failure_category": failure_category,
            "latency_seconds": latency,
            "latency_trace": state.get("latency_trace", {})
        }
        results.append(record)
        completed_keys.add(sample_key)

        # Lưu Checkpoint lũy tiến sau TỪNG câu
        try:
            checkpoint_data = {
                "benchmark_name": "Two-Stage Quantitative Benchmark (SQL + Python Quant)",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_size": sample_size,
                "completed_count": len(results),
                "quant_only": True,
                "results": results
            }
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Checkpoint] Không thể ghi file: {e}")

    total_bench_time = round(time.time() - start_bench_time, 2)

    # 5. Tổng hợp KPIs
    total_eval = len(results)
    total_correct = sum(1 for r in results if r["score"] == 1)
    accuracy_pct = (total_correct / total_eval * 100.0) if total_eval > 0 else 0.0

    sql_success_count = sum(1 for r in results if r.get("sql_success"))
    sql_success_rate = (sql_success_count / total_eval * 100.0) if total_eval > 0 else 0.0

    python_success_count = sum(1 for r in results if r.get("python_success"))
    python_success_rate = (python_success_count / total_eval * 100.0) if total_eval > 0 else 0.0

    total_latency = sum(r.get("latency_seconds", 0.0) for r in results)
    avg_latency = (total_latency / total_eval) if total_eval > 0 else 0.0

    # Phân tích các nguyên nhân lỗi
    failures = [r for r in results if r["score"] == 0]
    failure_counts: Dict[str, int] = {}
    for f in failures:
        cat = f.get("failure_category", "UNKNOWN")
        failure_counts[cat] = failure_counts.get(cat, 0) + 1

    sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

    # Cập nhật checkpoint file với summary hoàn chỉnh
    final_checkpoint_data = {
        "benchmark_name": "Two-Stage Quantitative Benchmark (SQL + Python Quant)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": sample_size,
        "total_evaluated": total_eval,
        "metrics": {
            "execution_accuracy_pct": round(accuracy_pct, 2),
            "sql_extraction_success_rate_pct": round(sql_success_rate, 2),
            "python_execution_success_rate_pct": round(python_success_rate, 2),
            "average_latency_seconds": round(avg_latency, 3),
            "total_benchmark_time_seconds": total_bench_time
        },
        "failure_mode_analysis": sorted_failures,
        "results": results
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(final_checkpoint_data, f, ensure_ascii=False, indent=2)

    # In báo cáo tổng kết ra terminal
    print("\n" + "="*80)
    print("🏆 BÁO CÁO TỔNG KẾT: 100-SAMPLE TWO-STAGE QUANTITATIVE BENCHMARK")
    print("="*80)
    print(f"📊 Tổng số mẫu kiểm thử:                {total_eval} / {sample_size}")
    print(f"🎯 Execution Accuracy (Exact Match <=1%): {accuracy_pct:.2f}% ({total_correct}/{total_eval})")
    print(f"📥 SQL Extraction Success Rate:         {sql_success_rate:.2f}% ({sql_success_count}/{total_eval})")
    print(f"🐍 Python Execution Success Rate:       {python_success_rate:.2f}% ({python_success_count}/{total_eval})")
    print(f"⚡ Độ trễ trung bình mỗi câu:           {avg_latency:.3f} giây / câu")
    print(f"⏱️ Tổng thời gian chạy benchmark:       {total_bench_time:.2f} giây")
    print("-" * 80)
    print("🔍 PHÂN TÍCH NGUYÊN NHÂN LỖI (TOP FAILURE MODES):")
    if sorted_failures:
        for rank, (cat, count) in enumerate(sorted_failures, start=1):
            pct = (count / len(failures)) * 100.0 if failures else 0.0
            print(f"  {rank}. {cat}: {count} câu ({pct:.1f}% tổng số lỗi)")
    else:
        print("  🎉 Không ghi nhận bất kỳ câu lỗi nào!")
    print("="*80 + "\n")

    return final_checkpoint_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chạy 100-sample Two-Stage Benchmark")
    parser.add_argument("--samples", type=int, default=100, help="Số mẫu benchmark (mặc định 100).")
    parser.add_argument("--checkpoint", type=str, default="docs/two_stage_100_pilot.json", help="Đường dẫn file checkpoint.")
    parser.add_argument("--fresh", action="store_true", help="Chạy lại từ đầu, bỏ qua checkpoint cũ.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để chọn mẫu.")
    args = parser.parse_args()

    run_benchmark(
        sample_size=args.samples,
        checkpoint_path=args.checkpoint,
        random_seed=args.seed,
        fresh=args.fresh
    )
