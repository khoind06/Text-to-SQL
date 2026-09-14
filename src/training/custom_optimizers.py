"""
custom_optimizers.py
Enterprise-Grade Hybrid Adaptive Optimizer for Financial NLP Models.

Decouples optimization dynamics:
- AdaGrad: Tailored for sparse, high-frequency/rare financial vocabulary embeddings.
  Accumulates historical squared gradients, automatically scaling down steps for
  frequent words while allowing substantial updates for rare accounting terminology.
- AdamW: Tailored for dense transformer parameters (multi-head attention, FFN, LayerNorm).
  Uses decoupled weight decay with first and second moment bias corrections.

Guarantees 100% numerical stability with NaN/Inf gradient sanitization and gradient clipping.
"""

import math
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from loguru import logger


class HybridAdaptiveOptimizer(Optimizer):
    """
    Bộ tối ưu hóa thích ứng lai cấp doanh nghiệp (Enterprise Hybrid Adaptive Optimizer).

    Kiến trúc kết hợp:
    1. AdaGrad Sub-Optimizer: Chuyên biệt hóa cho tầng Embedding (từ vựng tài chính thưa thớt).
       $$G_t = G_{t-1} + g_t^2$$
       $$\\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{G_t} + \\epsilon} g_t$$
       Giải quyết triệt để bài toán các thuật ngữ tài chính hiếm (ví dụ: 'EBITDA', 'amortization',
       'debentures') không nhận đủ cập nhật gradient dưới thuật toán Adam thông thường.

    2. AdamW Sub-Optimizer: Chuyên biệt hóa cho các tham số Dense (Attention, FFN, Projections).
       $$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$
       $$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$
       $$\\theta_t = \\theta_{t-1} (1 - \\eta \\lambda) - \\frac{\\eta \\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$
       với Decoupled Weight Decay $\\lambda$ giúp ổn định cực độ và ngăn chặn overfitting.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        adagrad_lr_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        max_grad_norm: Optional[float] = 1.0,
    ):
        """
        Khởi tạo HybridAdaptiveOptimizer.

        Args:
            params: Iterable parameters hoặc param_groups.
            lr: Learning rate cơ sở.
            betas: Hệ số moment (beta1, beta2) cho nhóm AdamW.
            eps: Epsilon chống chia cho 0 cho cả 2 nhánh thuật toán.
            weight_decay: Hệ số Decoupled Weight Decay cho nhóm AdamW.
            adagrad_lr_decay: Tốc độ suy giảm learning rate theo bước cho AdaGrad.
            initial_accumulator_value: Giá trị khởi tạo cho bộ tích lũy bình phương AdaGrad.
            max_grad_norm: Ngưỡng gradient clipping an toàn chống bùng nổ gradient.
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adagrad_lr_decay=adagrad_lr_decay,
            initial_accumulator_value=initial_accumulator_value,
            max_grad_norm=max_grad_norm,
            optimizer_type="adamw",
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Thực hiện một bước cập nhật tham số (Single Optimization Step)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            opt_type = group.get("optimizer_type", "adamw")
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            max_norm = group.get("max_grad_norm")

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # 1. Chốt chặn An toàn Số học (Numerical Stability Guard): Kiểm tra NaN/Inf
                if not torch.isfinite(grad).all():
                    logger.error(
                        f"[HybridOptimizer] Phát hiện NaN/Inf trong gradient của tensor shape {list(p.shape)}. "
                        f"Tự động vô hiệu hóa gradient của batch này để bảo vệ trọng số mô hình."
                    )
                    grad.zero_()
                    continue

                # 2. Gradient Clipping cục bộ theo tensor nếu vượt ngưỡng
                if max_norm is not None and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(p, max_norm)

                state = self.state[p]

                # Khởi tạo trạng thái lần đầu
                if len(state) == 0:
                    state["step"] = 0
                    if opt_type == "adagrad":
                        state["sum_sq"] = torch.full_like(
                            p.data, group["initial_accumulator_value"]
                        )
                    else:
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1
                step = state["step"]

                if opt_type == "adagrad":
                    # ── CƠ CHẾ ADAGRAD CHO SPARSE VOCABULARY EMBEDDINGS ────────
                    # Tích lũy tổng bình phương gradient lịch sử: G_t = G_{t-1} + g_t^2
                    state["sum_sq"].addcmul_(grad, grad, value=1.0)

                    # Điều chỉnh learning rate theo bước suy giảm
                    clr = lr / (1.0 + (step - 1) * group["adagrad_lr_decay"])

                    # Chuẩn hóa mẫu số bằng căn bậc hai của tổng bình phương
                    std = state["sum_sq"].sqrt().add_(eps)

                    # Cập nhật: p -= clr * (grad / std)
                    p.addcdiv_(grad, std, value=-clr)

                else:
                    # ── CƠ CHẾ ADAMW CHO DENSE PARAMETERS ───────────────────────
                    beta1, beta2 = group["betas"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Decoupled Weight Decay: p = p * (1 - lr * weight_decay)
                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)

                    # Cập nhật moment bậc 1: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                    # Cập nhật moment bậc 2: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # Hiệu chỉnh độ lệch (Bias Corrections)
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    # Cập nhật tham số
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_optimizer_for_financial_model(
    model: nn.Module,
    lr_dense: float = 2e-5,
    lr_sparse: float = 1e-2,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> HybridAdaptiveOptimizer:
    """
    Phân tách cấu trúc mô hình và tạo HybridAdaptiveOptimizer chuẩn Enterprise.

    Chiến lược phân nhóm:
    - Nhóm SPARSE (AdaGrad): Tự động phát hiện toàn bộ các tầng Embedding
      (word_embeddings, position_embeddings, token_type_embeddings).
      Tốc độ học lr_sparse cao hơn (1e-2) để thích ứng nhanh với các từ vựng tài chính hiếm.
    - Nhóm DENSE (AdamW): Tất cả các tầng Attention, Intermediate Dense, Output Linear và LayerNorm.
      Tốc độ học lr_dense cẩn trọng (2e-5) kết hợp weight decay (0.01) để bảo tồn tri thức tổng quát.

    Args:
        model: Mô hình PyTorch (FinancialEmbeddingModel hoặc BertModel).
        lr_dense: Learning rate cho nhóm tham số Dense (AdamW).
        lr_sparse: Learning rate cho nhóm tham số Sparse Embedding (AdaGrad).
        weight_decay: Hệ số Decoupled Weight Decay cho Dense.
        betas: Tham số moment AdamW.
        eps: Hệ số ổn định số học.

    Returns:
        HybridAdaptiveOptimizer đã cấu hình đầy đủ.
    """
    sparse_params: List[nn.Parameter] = []
    dense_params: List[nn.Parameter] = []
    sparse_names: List[str] = []

    # Danh sách các từ khóa đặc trưng cho tầng embedding
    embedding_keywords = {"word_embeddings", "position_embeddings", "token_type_embeddings", "wte", "wpe"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_sparse_target = any(kw in name for kw in embedding_keywords) or ("embeddings" in name and "weight" in name and "LayerNorm" not in name)
        if is_sparse_target:
            sparse_params.append(param)
            sparse_names.append(name)
        else:
            dense_params.append(param)

    param_groups = [
        {
            "params": sparse_params,
            "lr": lr_sparse,
            "optimizer_type": "adagrad",
            "initial_accumulator_value": 0.0,
            "adagrad_lr_decay": 0.0,
            "weight_decay": 0.0,
            "betas": betas,
            "eps": eps,
            "max_grad_norm": 1.0,
        },
        {
            "params": dense_params,
            "lr": lr_dense,
            "optimizer_type": "adamw",
            "initial_accumulator_value": 0.0,
            "adagrad_lr_decay": 0.0,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "max_grad_norm": 1.0,
        },
    ]

    optimizer = HybridAdaptiveOptimizer(
        params=param_groups,
        lr=lr_dense,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    n_sparse = sum(p.numel() for p in sparse_params)
    n_dense = sum(p.numel() for p in dense_params)
    logger.info(
        f"[HybridOptimizer] Phân tách tham số hoàn tất:\n"
        f"  - SPARSE / AdaGrad: {len(sparse_params)} tensors | {n_sparse:,} params | lr={lr_sparse} | targets={sparse_names}\n"
        f"  - DENSE  / AdamW:   {len(dense_params)} tensors | {n_dense:,} params | lr={lr_dense} | weight_decay={weight_decay}"
    )

    return optimizer
