"""
trainer.py
FinQATrainer: Standard PyTorch training loop with Cosine Embedding Loss,
Early Stopping, and Loguru logging for domain-specific financial embedding fine-tuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from loguru import logger


@dataclass
class TrainingConfig:
    """Configuration dataclass for FinQATrainer."""

    num_epochs: int = 10
    batch_size: int = 16
    lr_dense: float = 2e-5
    lr_sparse: float = 1e-2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 10
    val_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopping:
    """
    Early Stopping monitor: dừng huấn luyện nếu validation loss không giảm
    sau `patience` epochs liên tiếp.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            logger.info(f"[EarlyStopping] Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            logger.warning(
                f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs. "
                f"Best: {self.best_loss:.6f}, Current: {val_loss:.6f}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.warning(
                    "[EarlyStopping] Triggered! Training halted to prevent overfitting."
                )
        return self.should_stop


class CosineSimilarityLoss(nn.Module):
    """
    Contrastive Cosine Embedding Loss cho bài toán Sentence Embedding.
    Với các cặp (anchor, positive): loss = 1 - cos_sim(anchor, positive).
    Với các cặp (anchor, negative): loss = max(0, cos_sim(anchor, negative) - margin).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction="mean")

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings_a: (batch_size, dim) -- anchor embeddings
            embeddings_b: (batch_size, dim) -- positive/negative embeddings
            labels: (batch_size,) -- +1 for similar, -1 for dissimilar

        Returns:
            Scalar loss tensor
        """
        return self.loss_fn(embeddings_a, embeddings_b, labels)


class FinQATrainer:
    """
    Vòng lặp huấn luyện chuẩn PyTorch cho FinancialEmbeddingModel trên dữ liệu FinQA.
    Bao gồm:
    - Forward pass + Cosine Embedding Loss computation
    - Backward pass + Gradient clipping
    - HybridAdaptiveOptimizer step (AdamW cho dense, AdaGrad cho sparse embeddings)
    - Validation evaluation mỗi epoch
    - Early Stopping sau `patience` epochs không cải thiện
    - Loguru logging chi tiết theo từng batch và epoch
    - Checkpoint lưu mô hình tốt nhất
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or CosineSimilarityLoss(margin=0.5)
        self.device = self.config.device
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)

        # Move model and loss to device
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Checkpoint directory
        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        logger.info(
            f"[Trainer] Initialized on device={self.device} | "
            f"epochs={self.config.num_epochs} | "
            f"early_stop_patience={self.config.early_stopping_patience}"
        )

    def _forward_batch(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass for one batch.
        Expects batch = (input_ids_a, attn_a, input_ids_b, attn_b, labels)
        """
        input_ids_a, attn_a, input_ids_b, attn_b, labels = [
            t.to(self.device) for t in batch
        ]

        emb_a = self.model(input_ids=input_ids_a, attention_mask=attn_a)
        emb_b = self.model(input_ids=input_ids_b, attention_mask=attn_b)
        loss = self.loss_fn(emb_a, emb_b, labels)
        return loss, (emb_a, emb_b)

    def train_epoch(self, epoch: int) -> float:
        """Single training epoch. Returns mean training loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()
            loss, _ = self._forward_batch(batch)

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"[Trainer] NaN/Inf loss detected at epoch={epoch} step={step}. "
                    "Skipping batch."
                )
                continue

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            total_loss += loss.item()

            if step % self.config.log_every_n_steps == 0 or step == n_batches:
                logger.info(
                    f"[Train] Epoch {epoch}/{self.config.num_epochs} | "
                    f"Step {step}/{n_batches} | Loss={loss.item():.6f}"
                )

        mean_loss = total_loss / max(n_batches, 1)
        return mean_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """Validation evaluation. Returns mean validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)

        for batch in self.val_loader:
            loss, _ = self._forward_batch(batch)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save best model checkpoint."""
        ckpt_path = self.ckpt_dir / f"financial_embedding_epoch{epoch}_val{val_loss:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        logger.success(f"[Trainer] Checkpoint saved: {ckpt_path}")

    def train(self) -> Dict[str, List[float]]:
        """
        Vòng lặp huấn luyện đầy đủ:
        1. Train epoch -> tính train_loss
        2. Evaluate -> tính val_loss
        3. Ghi log epoch
        4. Lưu checkpoint nếu là val_loss tốt nhất
        5. Kiểm tra Early Stopping
        """
        best_val_loss = float("inf")
        logger.info("[Trainer] Starting training loop...")

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            logger.info(
                f"[Trainer] Epoch {epoch}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Save checkpoint if best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

            # Check early stopping
            if self.early_stopping(val_loss):
                logger.warning(
                    f"[Trainer] Early stopping triggered at epoch {epoch}. "
                    f"Best val loss: {self.early_stopping.best_loss:.6f}"
                )
                break

        logger.success(
            f"[Trainer] Training complete. Best val loss: {best_val_loss:.6f}"
        )
        return self.history
