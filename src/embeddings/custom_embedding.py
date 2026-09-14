"""
custom_embedding.py
Domain-Specific Financial Embedding Model with Depth-Aware Initialization
and Mean Pooling for dense financial retrieval and multi-agent context.
"""

from typing import List, Optional, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig, 
    BertTokenizer, 
    BertConfig, 
    BertModel
)
from loguru import logger

from src.embeddings.layer_initialization import DepthAwareInitializer


class FinancialEmbeddingModel(nn.Module):
    """
    Mô hình Embedding chuyên biệt cho dữ liệu tài chính (Financial Domain-Specific Embedding).
    Tích hợp bộ khởi tạo DepthAwareInitializer để tối ưu hóa tỷ lệ khởi tạo theo độ sâu tầng mạng,
    và cơ chế Mean Pooling có bù trừ Attention Mask.
    """

    def __init__(
        self,
        model_name_or_path: str = "yiyanghkust/finbert-tone",
        apply_depth_scaling: bool = True,
        depth_scale_factor: float = 1.0,
        normalize_embeddings: bool = True,
        device: Optional[str] = None
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self._target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Đang tải pre-trained backbone: '{model_name_or_path}'...")
        
        # Load Config & Backbone (hỗ trợ tương thích ngược cho FinBERT trên transformers mới)
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        except Exception as e:
            logger.warning(f"AutoConfig fallback: {e}. Sử dụng BertConfig trực tiếp.")
            config = BertConfig.from_pretrained(model_name_or_path)
            self.backbone = BertModel.from_pretrained(model_name_or_path, config=config)

        # Load Tokenizer (fallback sang BertTokenizer nếu AutoTokenizer yêu cầu)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            logger.warning(f"AutoTokenizer fallback: {e}. Sử dụng BertTokenizer trực tiếp.")
            self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        self.hidden_dimension = getattr(config, "hidden_size", 768)

        # Áp dụng DepthAwareInitializer
        self.initializer_report = {}
        if apply_depth_scaling:
            logger.info("Áp dụng DepthAwareInitializer lên các tầng transformer...")
            self.initializer = DepthAwareInitializer(
                scale_factor=depth_scale_factor,
                use_deepnet_scale=True
            )
            self.initializer_report = self.initializer.apply(self.backbone)

        self.to(self._target_device)
        logger.success(f"FinancialEmbeddingModel sẵn sàng trên thiết bị: {self._target_device}")

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dimension

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Thực hiện Mean Pooling trên các token embeddings, loại trừ các padding tokens.
        
        Args:
            token_embeddings: Tensor có shape (batch_size, seq_len, hidden_dim)
            attention_mask: Tensor có shape (batch_size, seq_len)

        Returns:
            Sentence embeddings có shape (batch_size, hidden_dim)
        """
        # Mở rộng attention_mask sang kích thước (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Tính tổng vector token có tính trọng số mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # Tính số lượng token hợp lệ (clamp tối thiểu 1e-9 để tránh chia cho 0)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # Trung bình cộng
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass trả về sentence embedding đại diện cho chuỗi văn bản tài chính.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len) tùy chọn

        Returns:
            embeddings: (batch_size, hidden_dimension)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        sentence_embeddings = self._mean_pooling(token_embeddings, attention_mask)

        if self.normalize_embeddings:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Tiện ích mã hóa danh sách các đoạn văn bản tài chính sang vector embeddings.
        Quản lý bộ nhớ GPU theo chuẩn PyTorch Caching Allocator:
        Tuyệt đối KHÔNG gọi empty_cache() định kỳ gây nghẽn cổ chai (I/O sync stall).
        Chỉ can thiệp khi bắt gặp ngoại lệ OutOfMemoryError.
        """
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    encoded_inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    ).to(self._target_device)

                    batch_emb = self.forward(**encoded_inputs)
                    all_embeddings.append(batch_emb.cpu())

                except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                    if "out of memory" in str(err).lower():
                        logger.critical(
                            f"[GPU Memory Guard] OOM detected at batch {i//batch_size}. "
                            f"Kích hoạt khẩn cấp torch.cuda.empty_cache() và thử lại với batch nhỏ hơn."
                        )
                        torch.cuda.empty_cache()
                        # Xử lý tuần tự từng mẫu nếu batch bị OOM
                        for single_text in batch_texts:
                            inp = self.tokenizer(
                                [single_text],
                                padding=True,
                                truncation=True,
                                max_length=max_length,
                                return_tensors="pt"
                            ).to(self._target_device)
                            single_emb = self.forward(**inp)
                            all_embeddings.append(single_emb.cpu())
                    else:
                        raise err

        return torch.cat(all_embeddings, dim=0)
