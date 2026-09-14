"""
scripts/smoke_test_5_samples.py
Chạy thử nghiệm khói (Smoke Test) trên 5 mẫu ngẫu nhiên từ data/evaluation_ground_truth.json.
Kiểm tra chi tiết:
- Routing bảng thực tế
- SQL do Agent 1 sinh (có dính financial_data không)
- Dữ liệu trả về từ SQLite & giá trị bóc tách
- Thẩm định độ chính xác (math.isclose 1% tolerance)
- Độ trễ (latency)
"""

import json
import random
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.evaluation_pipeline import FinQAEvaluator
from src.agents.orchestrator import FinancialOrchestrator
from loguru import logger


def run_smoke_test(num_samples: int = 5, seed: int = 42):
    gt_path = Path("data/evaluation_ground_truth.json")
    if not gt_path.exists():
        logger.error(f"Không tìm thấy file: {gt_path}")
        return

    with open(gt_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # Chọn 5 mẫu ngẫu nhiên có đáp án hợp lệ
    valid_data = [d for d in all_data if d.get("exe_ans") and str(d.get("exe_ans")).strip()]
    random.seed(seed)
    selected_samples = random.sample(valid_data, min(num_samples, len(valid_data)))

    logger.info(f"Khởi tạo Orchestrator & Evaluator cho Smoke Test ({num_samples} mẫu)...")
    evaluator = FinQAEvaluator(sample_size=num_samples)

    report_items = []

    print("\n" + "="*80)
    print(f"🔥 BẮT ĐẦU SMOKE TEST: {num_samples} MẪU NGẪU NHIÊN (ZERO-TRUST VALIDATION)")
    print("="*80 + "\n")

    for i, sample in enumerate(selected_samples, 1):
        q = sample["question"]
        gold = str(sample["exe_ans"]).strip()
        table_id = sample.get("table_id", "")
        sample_id = sample.get("sample_id", "")

        eval_q = f"In {table_id}: {q}" if table_id and table_id.lower() not in q.lower() else q

        t0 = time.perf_counter()
        try:
            state = evaluator.orchestrator.run(eval_q)
            latency = time.perf_counter() - t0
        except Exception as e:
            latency = time.perf_counter() - t0
            state = {"sql_query": "", "sql_result": None, "analysis_report": f"Error: {e}"}

        sql_query = state.get("sql_query") or ""
        sql_result = state.get("sql_result")
        analysis_report = state.get("analysis_report") or ""

        # Kiểm tra dính chữ financial_data
        has_hallucinated_table = "financial_data" in sql_query.lower()

        # Số dòng trả về
        row_count = len(sql_result) if hasattr(sql_result, "__len__") else (0 if sql_result is None else 1)

        # Đánh giá bằng strict evaluate_accuracy
        score, reason = evaluator.evaluate_accuracy(
            sql_result=sql_result,
            gold_answer=gold,
            analysis_report=analysis_report
        )

        item_res = {
            "sample_num": i,
            "sample_id": sample_id,
            "table_id": table_id,
            "question": q,
            "gold_answer": gold,
            "sql_query": sql_query,
            "has_hallucinated_financial_data": has_hallucinated_table,
            "row_count": row_count,
            "score": score,
            "reason": reason,
            "latency_seconds": round(latency, 3),
            "analysis_snippet": analysis_report[:120] if analysis_report else ""
        }
        report_items.append(item_res)

        status_tag = "✅ PASS" if score == 1 else "❌ FAIL"
        hallucination_tag = "❌ CÓ DÍNH (LỖI)" if has_hallucinated_table else "✅ SẠCH (KHÔNG DÍNH)"

        print(f"--- [MẪU {i}/{num_samples}] {status_tag} (Độ trễ: {latency:.2f}s) ---")
        print(f"• Question đầu vào:             {q}")
        print(f"• Bảng thực tế đưa vào Schema:   {table_id}")
        print(f"• SQL Query Agent 1 sinh:        {sql_query}")
        print(f"• Kiểm tra 'financial_data':     {hallucination_tag}")
        print(f"• Số dòng trả về từ SQLite:     {row_count} dòng")
        print(f"• Đáp án vàng (exe_ans):        {gold}")
        print(f"• Kết quả đối soát (Tolerance):  {reason}")
        print()

    # Thống kê tổng hợp
    passed_count = sum(1 for item in report_items if item["score"] == 1)
    avg_latency = sum(item["latency_seconds"] for item in report_items) / len(report_items)
    any_hallucination = any(item["has_hallucinated_financial_data"] for item in report_items)

    print("="*80)
    print("📊 TỔNG HỢP KẾT QUẢ SMOKE TEST:")
    print(f"- Số mẫu vượt qua (Score = 1):       {passed_count} / {num_samples} ({passed_count/num_samples*100:.1f}%)")
    print(f"- Độ trễ trung bình:                {avg_latency:.3f} giây / câu")
    print(f"- Bị dính bảng giả 'financial_data': {'CÓ PHÁT HIỆN LỖI' if any_hallucination else 'HOÀN TOÀN KHÔNG BỊ (0%)'}")
    print("="*80)

    with open("docs/smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump(report_items, f, ensure_ascii=False, indent=2)
    logger.success("Đã lưu chi tiết kết quả Smoke Test ra: docs/smoke_test_results.json")


if __name__ == "__main__":
    run_smoke_test(num_samples=5, seed=42)
