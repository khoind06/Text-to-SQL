"""
test_embeddings.py
Unit tests for:
1. DepthAwareInitializer: formula correctness + weight scaling on mock layers
2. FinancialEmbeddingModel: forward pass, mean pooling, embedding shape
3. HybridAdaptiveOptimizer: dummy batch backward pass -- ensures no NaN gradient
4. Optimizer parameter group separation (sparse vs dense)
5. EarlyStopping logic
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.embeddings.layer_initialization import DepthAwareInitializer
from src.embeddings.custom_embedding import FinancialEmbeddingModel
from src.training.custom_optimizers import (
    HybridAdaptiveOptimizer,
    build_optimizer_for_financial_model,
)
from src.training.trainer import (
    EarlyStopping,
    CosineSimilarityLoss,
    FinQATrainer,
    TrainingConfig,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def finbert_model():
    """Load FinancialEmbeddingModel once for all tests."""
    model = FinancialEmbeddingModel(
        model_name_or_path="yiyanghkust/finbert-tone",
        apply_depth_scaling=True,
        normalize_embeddings=True,
        device="cpu",
    )
    return model


@pytest.fixture(scope="module")
def hybrid_optimizer(finbert_model):
    """Build HybridAdaptiveOptimizer for FinancialEmbeddingModel."""
    return build_optimizer_for_financial_model(
        model=finbert_model,
        lr_dense=2e-5,
        lr_sparse=1e-2,
        weight_decay=0.01,
    )


# ======================================================================
# TEST 1: DepthAwareInitializer formula correctness
# ======================================================================

def test_depth_aware_initializer_formula():
    """Scale phải giảm đơn điệu: scale(L=1) > scale(L=4) > scale(L=12)."""
    initializer = DepthAwareInitializer(scale_factor=1.0, use_deepnet_scale=True)
    s1 = initializer.compute_layer_scale(1)
    s4 = initializer.compute_layer_scale(4)
    s12 = initializer.compute_layer_scale(12)

    assert s1 == pytest.approx(1.0 / math.sqrt(2.0 * 1), rel=1e-5)
    assert s4 == pytest.approx(1.0 / math.sqrt(2.0 * 4), rel=1e-5)
    assert s12 == pytest.approx(1.0 / math.sqrt(2.0 * 12), rel=1e-5)
    assert s1 > s4 > s12


# ======================================================================
# TEST 2: DepthAwareInitializer on mock transformer layers
# ======================================================================

def test_depth_aware_initializer_on_mock_layers():
    """Kiểm tra trọng số được thu nhỏ sau khi áp dụng DepthAwareInitializer."""
    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([
                nn.ModuleDict({
                    "attention": nn.ModuleDict({
                        "output": nn.ModuleDict({"dense": nn.Linear(64, 64)})
                    }),
                    "output": nn.ModuleDict({"dense": nn.Linear(64, 64)})
                }) for _ in range(4)
            ])

    mock_model = MockTransformer()
    orig_std_l1 = mock_model.encoder.layer[0].attention.output.dense.weight.std().item()
    orig_std_l4 = mock_model.encoder.layer[3].attention.output.dense.weight.std().item()

    initializer = DepthAwareInitializer(scale_factor=1.0)
    report = initializer.apply(mock_model)

    assert report["total_layers"] == 4
    assert report["scaled_weights"] == 8  # 2 modules x 4 layers

    new_std_l1 = mock_model.encoder.layer[0].attention.output.dense.weight.std().item()
    new_std_l4 = mock_model.encoder.layer[3].attention.output.dense.weight.std().item()

    assert new_std_l1 < orig_std_l1
    assert new_std_l4 < orig_std_l4
    # Tầng sâu hơn phải được thu nhỏ nhiều hơn
    assert (new_std_l4 / orig_std_l4) < (new_std_l1 / orig_std_l1)


# ======================================================================
# TEST 3: FinancialEmbeddingModel forward pass + shape
# ======================================================================

def test_financial_embedding_model_forward(finbert_model):
    """Kiểm tra output shape (2, 768) và L2-norm ≈ 1.0."""
    texts = [
        "Revenue increased 12% year-over-year.",
        "Credit risk exposure is elevated in commercial real estate.",
    ]
    embeddings = finbert_model.encode(texts)

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 768)
    # Kiểm tra L2-norm = 1.0 do normalize_embeddings=True
    assert torch.norm(embeddings[0]).item() == pytest.approx(1.0, abs=1e-4)
    assert torch.norm(embeddings[1]).item() == pytest.approx(1.0, abs=1e-4)


# ======================================================================
# TEST 4: Mean pooling with attention mask
# ======================================================================

def test_mean_pooling_attention_mask(finbert_model):
    """Kiểm tra mean pooling bỏ qua đúng các padding token."""
    token_embs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])  # token thứ 3 là padding

    # Sử dụng model tạm thời với số chiều nhỏ để test hàm _mean_pooling
    pooled = finbert_model._mean_pooling(token_embs, mask)
    expected = torch.tensor([[2.0, 3.0]])
    assert torch.allclose(pooled, expected, atol=1e-5)


# ======================================================================
# TEST 5: HybridAdaptiveOptimizer - Backward pass không có NaN gradient
# ======================================================================

def test_hybrid_optimizer_no_nan_gradient(finbert_model, hybrid_optimizer):
    """
    [CRITICAL] Đưa dummy batch qua mô hình, backward, rồi kiểm tra:
    1. Không có gradient nào là NaN hoặc Inf.
    2. Sau optimizer.step(), tham số đã được cập nhật (không đứng yên).
    """
    finbert_model.train()

    # Tạo dummy token tensors giả lập (batch_size=2, seq_len=16)
    vocab_size = finbert_model.tokenizer.vocab_size
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Snapshot trọng số trước khi backward
    params_before = [p.clone().detach() for p in finbert_model.parameters() if p.requires_grad]

    # Forward pass
    emb = finbert_model(input_ids=input_ids, attention_mask=attention_mask)

    # Tính loss đơn giản: mean của tất cả embedding values
    loss = emb.mean()

    # Backward pass
    hybrid_optimizer.zero_grad()
    loss.backward()

    # Kiểm tra gradient không có NaN/Inf
    nan_found = False
    for name, param in finbert_model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_found = True
                pytest.fail(
                    f"NaN/Inf gradient detected in param: {name}"
                )

    assert not nan_found, "NaN hoặc Inf gradient được phát hiện trong quá trình backward!"

    # Optimizer step
    hybrid_optimizer.step()

    # Kiểm tra tham số đã thực sự được cập nhật
    params_after = [p.clone().detach() for p in finbert_model.parameters() if p.requires_grad]
    any_updated = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert any_updated, "Không có tham số nào được cập nhật sau optimizer.step()!"

    finbert_model.eval()


# ======================================================================
# TEST 6: Optimizer parameter group separation
# ======================================================================

def test_optimizer_param_group_separation(finbert_model, hybrid_optimizer):
    """Kiểm tra 2 nhóm tham số: sparse (AdaGrad) và dense (AdamW)."""
    param_groups = hybrid_optimizer.param_groups
    assert len(param_groups) == 2

    sparse_group = param_groups[0]
    dense_group = param_groups[1]

    assert sparse_group["optimizer_type"] == "adagrad"
    assert dense_group["optimizer_type"] == "adamw"

    # Sparse group phải có ít nhất 1 embedding tensor (word_embeddings)
    assert len(sparse_group["params"]) >= 1

    # Dense group phải có nhiều tham số hơn
    assert len(dense_group["params"]) > len(sparse_group["params"])

    # Learning rate sparse > dense
    assert sparse_group["lr"] > dense_group["lr"]


# ======================================================================
# TEST 7: EarlyStopping logic
# ======================================================================

def test_early_stopping_triggers_after_patience():
    """Kiểm tra EarlyStopping dừng đúng sau `patience` epochs không cải thiện."""
    stopper = EarlyStopping(patience=3, min_delta=1e-4)

    # Epoch 1: cải thiện
    assert stopper(0.5) is False
    assert stopper.counter == 0

    # Epoch 2, 3, 4: không cải thiện
    assert stopper(0.52) is False
    assert stopper.counter == 1

    assert stopper(0.51) is False
    assert stopper.counter == 2

    # Epoch 5: trigger
    result = stopper(0.50)
    assert result is True
    assert stopper.should_stop is True


def test_early_stopping_resets_on_improvement():
    """Kiểm tra counter reset khi có cải thiện."""
    stopper = EarlyStopping(patience=3, min_delta=1e-4)
    stopper(0.5)    # best=0.5, counter=0
    stopper(0.52)   # counter=1
    stopper(0.4)    # improvement -> best=0.4, counter reset to 0
    assert stopper.counter == 0
    assert stopper.should_stop is False


# ======================================================================
# TEST 8: CosineSimilarityLoss output range
# ======================================================================

def test_cosine_similarity_loss():
    """Kiểm tra CosineSimilarityLoss trong khoảng [0, 2]."""
    loss_fn = CosineSimilarityLoss(margin=0.5)
    a = F.normalize(torch.randn(4, 128), dim=1)
    b = F.normalize(torch.randn(4, 128), dim=1)
    labels = torch.tensor([1, 1, -1, -1])

    loss = loss_fn(a, b, labels)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()
    assert 0.0 <= loss.item() <= 2.0


# Required import at module level for test 8
import torch.nn.functional as F
