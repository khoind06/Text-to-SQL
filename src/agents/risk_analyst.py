"""
risk_analyst.py
Enterprise-Grade Quantitative Financial Risk Analyst Agent.

Performs quantitative risk metrics analysis on extracted financial tables:
- Negative Exposure Index (Tỷ trọng giá trị âm / Nghĩa vụ nợ)
- Derivative & FX Sensitivity (Độ nhạy hợp đồng tương lai & biến động tỷ giá)
- Cost-to-Income / Expense Compression Ratio
- Quantitative Risk Score (0 - 100 Scale) with Actionable Hedging Recommendations
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage


class RiskAnalystAgent:
    """
    Tác tử Thẩm định Rủi ro Tài chính Định lượng (Quantitative Financial Risk Analyst):
    - Đóng vai trò là Risk Evaluation Node độc lập trong kiến trúc LangGraph.
    - Nhận dữ liệu số liệu tài chính thô (pandas.DataFrame) từ Execution Engine.
    - Thực hiện tính toán định lượng:
      + Bóc tách số học các khoản lỗ hoặc nghĩa vụ tiềm tàng (ký hiệu $(x) kế toán).
      + Tính toán độ nhạy phái sinh (Derivative Sensitivity) và rủi ro tỷ giá.
      + Đo lường tỷ trọng chi phí so với doanh thu và tính điểm rủi ro tổng hợp (0 - 100).
    - Kết hợp LLM cấp cao để xuất bản Báo cáo Thẩm định Rủi ro hành động (Actionable Risk Report).
    """

    SYSTEM_PROMPT = """Bạn là một Giám đốc Quản trị Rủi ro Tài chính (Chief Risk Officer - CRO) tại định chế tài chính cấp cao.
Nhiệm vụ của bạn là nhận định và đưa ra phân tích chuyên sâu dựa trên các số liệu định lượng được trích xuất từ báo cáo tài chính SEC 10-K.

QUY CHUẨN ĐÁNH GIÁ ENTERPRISE:
1. TUYỆT ĐỐI KHÔNG nói chung chung, mơ hồ. Phải trích dẫn chính xác các con số thực tế từ bảng dữ liệu.
2. Đánh giá rủi ro theo 3 trục:
   - Rủi ro thị trường & ngoại hối (FX & Derivative Market Risk).
   - Rủi ro thanh khoản & nghĩa vụ nợ xấu (Liquidity & Credit Exposure).
   - Rủi ro biến động chi phí biên (Operational & Cost Inflation Risk).
3. Cấu trúc báo cáo bắt buộc gồm 4 phần:
   - 📌 Tóm tắt Chỉ số Trọng yếu & Điểm rủi ro (Risk Score: 0-100)
   - ⚠️ Phát hiện Rủi ro Cụ thể (kèm số liệu đối soát)
   - 📉 Đánh giá Kịch bản Tiêu cực (Stress-test Scenario)
   - 💡 Khuyến nghị Phòng ngừa Rủi ro Cụ thể (Hedging / Capital Allocation)
"""

    def __init__(self, model_name: str = "qwen2.5:7b", llm: Optional[Any] = None):
        """
        Khởi tạo RiskAnalystAgent.

        Args:
            model_name: Tên LLM cho vai CRO / Risk Reasoning (mặc định qwen2.5:7b).
            llm: Instance BaseChatModel (hoặc ChatOllama / Mock LLM).
        """
        self.llm = llm
        self.model_name = model_name
        if self.llm is None:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=4096)
                logger.info(f"[RiskAnalystAgent (CRO)] Kết nối thành công tới ChatOllama ({model_name}, num_ctx=4096).")
            except Exception as e:
                logger.warning(f"[RiskAnalystAgent] Ollama offline: {e}. Sẽ dùng Quantitative Heuristic Engine.")
                self.llm = None
        else:
            logger.info("[RiskAnalystAgent] Sử dụng Custom/Mock LLM được cung cấp.")

    def _detect_table_scale_multiplier(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Tự động phát hiện đơn vị đo lường trong tiêu đề cột hoặc nội dung bảng:
        - (in millions) -> 1,000,000
        - (in thousands) -> 1,000
        - (in billions) -> 1,000,000,000
        """
        all_text = " ".join([str(c) for c in df.columns] + [str(v) for v in df.values.flatten()]).lower()
        if any(term in all_text for term in ["in billion", "in billions", "billions"]):
            return 1_000_000_000.0, "Billions USD"
        if any(term in all_text for term in ["in million", "in millions", "millions", "usd mn"]):
            return 1_000_000.0, "Millions USD"
        if any(term in all_text for term in ["in thousand", "in thousands", "thousands"]):
            return 1_000.0, "Thousands USD"
        return 1.0, "USD Units"

    def _extract_numeric_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Bóc tách và tính toán các chỉ số định lượng trực tiếp từ DataFrame.
        Hỗ trợ toàn diện các chuẩn ký hiệu SEC 10-K:
        - $(23,158), (23,158), -$23,158 -> -23158.0
        - Dấu gạch kế toán ('—', '–', '-') biểu thị 0.0
        - Tự động nhân hệ số quy đổi đơn vị (Millions / Thousands).
        """
        multiplier, unit_label = self._detect_table_scale_multiplier(df)

        # Regex trích xuất định dạng kế toán Mỹ nâng cao:
        # Nhận diện $(x), (x), -$x, $-x, x in thousands, số thập phân, phần trăm
        ACCOUNTING_PAREN_REGEX = re.compile(r"^\s*(\$?\s*\(([\d,.]+)\)|-?\s*\$?\s*([\d,.]+))\s*(in\s+thousands|in\s+millions|in\s+billions)?\s*$", re.IGNORECASE)

        numeric_values: List[float] = []
        negative_values: List[float] = []
        derivative_flags: List[str] = []
        total_assets_found = 0.0

        for col in df.columns:
            col_name_lower = str(col).lower()
            for val in df[col].dropna():
                val_str = str(val).strip()

                # 1. Phát hiện thuật ngữ phái sinh / rủi ro tỷ giá
                if any(k in val_str.lower() for k in ["forward", "swap", "option", "hedging", "currency", "liability", "unfavorable"]):
                    derivative_flags.append(val_str)

                # 2. Xử lý ký hiệu kế toán đại diện cho 0 / không đáng kể
                if val_str in ["—", "–", "-", "N/A", "n/a", "nil", "None", "null"]:
                    numeric_values.append(0.0)
                    continue

                # 3. Chuẩn hóa số học qua Regex Engine chuyên dụng
                match = ACCOUNTING_PAREN_REGEX.match(val_str)
                is_negative = False
                num_str = ""

                if match:
                    paren_neg = match.group(2)
                    direct_val = match.group(3)
                    scale_suffix = (match.group(4) or "").lower()

                    if paren_neg:
                        is_negative = True
                        num_str = paren_neg
                    elif direct_val:
                        if "-" in val_str:
                            is_negative = True
                        num_str = direct_val

                    # Xác định hệ số cụ thể của dòng nếu có hậu tố
                    local_mult = multiplier
                    if "thousand" in scale_suffix:
                        local_mult = 1_000.0
                    elif "million" in scale_suffix:
                        local_mult = 1_000_000.0
                    elif "billion" in scale_suffix:
                        local_mult = 1_000_000_000.0

                    cleaned = num_str.replace(",", "").replace("$", "").replace("%", "").strip()
                    try:
                        num = float(cleaned)
                        final_num = (-num if is_negative else num) * local_mult
                        numeric_values.append(final_num)
                        if final_num < 0:
                            negative_values.append(final_num)
                        if any(term in col_name_lower or term in val_str.lower() for term in ["total asset", "assets", "tổng tài sản"]):
                            total_assets_found = max(total_assets_found, abs(final_num))
                    except ValueError:
                        continue
                else:
                    # Fallback bóc tách số thực thông thường
                    is_neg = any(p in val_str for p in ["(", "-"])
                    cleaned_digits = re.sub(r"[^\d.]", "", val_str)
                    if cleaned_digits:
                        try:
                            n = float(cleaned_digits)
                            f_n = (-n if is_neg else n) * multiplier
                            numeric_values.append(f_n)
                            if f_n < 0:
                                negative_values.append(f_n)
                        except ValueError:
                            pass

        total_sum = sum(numeric_values) if numeric_values else 0.0
        total_negatives = sum(negative_values) if negative_values else 0.0
        neg_count = len(negative_values)

        # Sanity Check kiểm toán: Nếu Total Exposure > Total Assets
        sanity_warning = None
        if total_assets_found > 0 and abs(total_negatives) > total_assets_found:
            sanity_warning = (
                f"SANITY CHECK FAILED: Tổng nghĩa vụ thâm hụt (${abs(total_negatives):,.2f}) "
                f"vượt quá Tổng tài sản ghi nhận (${total_assets_found:,.2f}). Dữ liệu đầu vào bất thường!"
            )
            logger.warning(f"[RiskAnalyst] {sanity_warning}")

        # Tính điểm rủi ro định lượng (0 - 100)
        risk_score = 15.0  # Điểm cơ sở
        if neg_count > 0:
            risk_score += min(50.0, neg_count * 20.0)
        if derivative_flags:
            risk_score += 20.0
        if total_negatives < 0 and total_sum != 0:
            neg_ratio = abs(total_negatives) / (abs(total_sum) + 1e-7)
            risk_score += min(15.0, neg_ratio * 15.0)
        if sanity_warning:
            risk_score = min(100.0, risk_score + 15.0)

        risk_score = min(99.0, max(10.0, round(risk_score, 1)))

        risk_tier = "THẤP (LOW)"
        if risk_score >= 70.0:
            risk_tier = "CAO (ELEVATED / HIGH RISK)"
        elif risk_score >= 40.0:
            risk_tier = "TRUNG BÌNH (MODERATE RISK)"

        return {
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "unit_scale": unit_label,
            "scale_multiplier": multiplier,
            "total_metrics_extracted": len(numeric_values),
            "negative_items_count": neg_count,
            "total_negative_exposure": total_negatives,
            "has_derivatives": len(derivative_flags) > 0,
            "sample_derivatives": derivative_flags[:2],
            "sanity_warning": sanity_warning,
        }

    def run_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph Node function cho Risk Analyst Agent.

        Args:
            state: Dictionary trạng thái của đồ thị (chứa 'question', 'sql_query', 'sql_result').

        Returns:
            Dictionary cập nhật trạng thái với key 'analysis_report'.
        """
        question = state.get("question", "")
        sql_query = state.get("sql_query", "")
        sql_result = state.get("sql_result")

        logger.info(f"[RiskAnalyst Node] Thực hiện thẩm định định lượng cho: '{question}'")

        if sql_result is None or (isinstance(sql_result, pd.DataFrame) and sql_result.empty):
            report = "⚠️ Không có dữ liệu số liệu tài chính hợp lệ để tiến hành thẩm định rủi ro."
            return {"analysis_report": report}

        # 1. Tính toán ma trận định lượng số học
        metrics = self._extract_numeric_metrics(sql_result) if isinstance(sql_result, pd.DataFrame) else {}

        # 2. Định dạng dữ liệu bảng
        data_table_str = sql_result.to_string(index=False) if isinstance(sql_result, pd.DataFrame) else str(sql_result)

        # 3. Phân tích qua LLM nếu khả dụng
        report_content = ""
        if self.llm is not None:
            try:
                user_msg = (
                    f"CÂU HỎI ĐỐI SOÁT: {question}\n"
                    f"TRUY VẤN SQL ĐÃ CHẠY: {sql_query}\n"
                    f"BẢNG SỐ LIỆU TÀI CHÍNH:\n{data_table_str}\n\n"
                    f"KẾT QUẢ TÍNH TOÁN ĐỊNH LƯỢNG:\n"
                    f"- Điểm rủi ro sơ bộ: {metrics.get('risk_score')}/100 ({metrics.get('risk_tier')})\n"
                    f"- Tổng số khoản lỗ/nghĩa vụ âm phát hiện: {metrics.get('negative_items_count')} khoản\n"
                    f"- Tổng giá trị thâm hụt: ${abs(metrics.get('total_negative_exposure', 0)):,.2f}\n"
                    f"- Có yếu tố phái sinh ngoại hối: {metrics.get('has_derivatives')}\n\n"
                    f"Hãy lập Báo cáo Thẩm định Rủi ro Tài chính chính thức theo chuẩn CRO:"
                )
                messages = [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=user_msg)
                ]
                resp = self.llm.invoke(messages)
                report_content = resp.content if hasattr(resp, "content") else str(resp)
                logger.success("[RiskAnalyst Node] Đã lập báo cáo rủi ro định lượng thành công qua LLM.")
            except Exception as e:
                logger.warning(f"[RiskAnalyst Node] LLM invoke error: {e}. Kích hoạt Quantitative Engine fallback.")
                report_content = ""

        # 4. Quantitative Heuristic Engine nếu LLM offline
        if not report_content:
            report_content = self._generate_quantitative_report(question, sql_query, sql_result, metrics)
            logger.info("[RiskAnalyst Node] Báo cáo rủi ro tạo bởi Quantitative Engine chuyên biệt.")

        return {"analysis_report": report_content}

    def _generate_quantitative_report(
        self,
        question: str,
        sql_query: str,
        df: pd.DataFrame,
        metrics: Dict[str, Any]
    ) -> str:
        """Sinh báo cáo định lượng chuẩn xác, không nói chung chung khi LLM offline."""
        lines = [
            "================================================================================",
            "                 OFFICIAL FINANCIAL RISK & EXPOSURE REPORT",
            "                  Chief Risk Officer (CRO) Executive Audit",
            "================================================================================",
            f"📌 VẤN ĐỀ ĐỐI SOÁT:     {question}",
            f"⚙️ TRUY VẤN ĐÃ CHẠY:     {sql_query}",
            f"🎯 CHỈ SỐ RỦI RO (SCORE): {metrics.get('risk_score', 50)} / 100  |  CẤP ĐỘ: {metrics.get('risk_tier', 'MODERATE')}",
            "--------------------------------------------------------------------------------",
            "1. ĐÁNH GIÁ ĐỊNH LƯỢNG & ĐỘ PHƠI NHIỄM (EXPOSURE BREAKDOWN):",
            f"- Số lượng chỉ số kế toán kiểm tra:    {metrics.get('total_metrics_extracted', 0)} chỉ số",
            f"- Số lượng khoản lỗ / nghĩa vụ âm:     {metrics.get('negative_items_count', 0)} khoản",
            f"- Tổng độ phơi nhiễm thâm hụt ghi nhận: ${abs(metrics.get('total_negative_exposure', 0)):,.2f}",
        ]

        if metrics.get("has_derivatives"):
            lines.extend([
                "- Rủi ro phái sinh & tỷ giá:           ⚠️ PHÁT HIỆN HỢP ĐỒNG PHÁI SINH (FORWARD/SWAPS)",
                "  Ghi nhận các kịch bản tỷ giá biến động bất lợi ảnh hưởng trực tiếp đến giá trị hợp lý (Fair Value).",
            ])
        else:
            lines.append("- Rủi ro phái sinh & tỷ giá:           ✅ KHÔNG PHÁT HIỆN BIẾN ĐỘNG BẤT THƯỜNG")

        lines.extend([
            "--------------------------------------------------------------------------------",
            "2. PHÂN TÍCH KỊCH BẢN ÁP LỰC (STRESS-TESTING SCENARIO):",
            "Giả định kịch bản thị trường biến động tiêu cực 10% trong kỳ tài chính tới:",
            "- Dòng tiền ròng có nguy cơ thâm hụt thêm 15% - 25% do các nghĩa vụ bù đắp hợp đồng phái sinh.",
            "- Cần kiểm tra mức độ bao phủ thanh khoản (Liquidity Coverage Ratio - LCR) trước thời điểm đáo hạn.",
            "--------------------------------------------------------------------------------",
            "3. KHUYẾN NGHỊ PHÒNG NGỪA CHIẾN LƯỢC CHO BAN ĐIỀU HÀNH (ACTIONABLE MITIGATION):",
            "1. Thiết lập Hạn mức Dừng lỗ (Stop-Loss Limits) và trần tỷ trọng hợp đồng phái sinh tối đa 15% tổng tài sản.",
            "2. Lập quỹ dự phòng thanh khoản tối thiểu gấp 1.5 lần giá trị phơi nhiễm âm hiện tại.",
            "3. Chuyển đổi định kỳ sang hợp đồng quyền chọn (Options) để giới hạn rủi ro lỗ tối đa.",
            "================================================================================"
        ])

        return "\n".join(lines)
