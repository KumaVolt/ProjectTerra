"""Vision encoder for Terra multimodal model.

A small ViT-style encoder that maps image patches into the same embedding space
as the text transformer. Can be trained independently or jointly.

Architecture: ViT-Tiny/Small with projection MLP into Terra's hidden_size.
- Patch embedding (conv2d)
- Transformer encoder blocks
- MLP projection → Terra hidden_size

This runs in parallel with text pre-training — the vision encoder can be
pre-trained on image-text contrastive loss (like SigLIP/CLIP) independently.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    """Vision encoder configuration."""

    image_size: int = 384
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 384  # ViT-Small
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    # Projection to LLM space
    projection_size: int = 896  # Must match TerraConfig.hidden_size

    @classmethod
    def vision_tiny(cls, projection_size: int = 896) -> "VisionConfig":
        """~6M params - fast experiments."""
        return cls(
            image_size=224,
            hidden_size=192,
            intermediate_size=768,
            num_hidden_layers=6,
            num_attention_heads=3,
            projection_size=projection_size,
        )

    @classmethod
    def vision_small(cls, projection_size: int = 896) -> "VisionConfig":
        """~22M params - good quality/speed tradeoff."""
        return cls(
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            projection_size=projection_size,
        )

    @classmethod
    def vision_base(cls, projection_size: int = 896) -> "VisionConfig":
        """~86M params - high quality."""
        return cls(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            projection_size=projection_size,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "VisionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def param_count_estimate(self) -> int:
        patch_embed = self.num_channels * (self.patch_size ** 2) * self.hidden_size
        pos_embed = (self.num_patches + 1) * self.hidden_size  # +1 for CLS
        attn = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O
        ffn = 2 * self.hidden_size * self.intermediate_size
        per_layer = attn + ffn + 2 * self.hidden_size  # + norms
        projection = self.hidden_size * self.projection_size * 2  # 2-layer MLP
        return patch_embed + pos_embed + self.num_hidden_layers * per_layer + projection


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings using a conv2d."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.position_embedding = nn.Embedding(config.num_patches + 1, config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, C, H, W)
        batch_size = pixel_values.shape[0]

        # Patch projection: (B, hidden, H/P, W/P) -> (B, num_patches, hidden)
        patches = self.projection(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patches], dim=1)

        # Add position embeddings
        seq_len = embeddings.shape[1]
        position_ids = torch.arange(seq_len, device=embeddings.device)
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        x = attn.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionProjection(nn.Module):
    """2-layer MLP that maps vision features → LLM embedding space."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.projection_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(config.projection_size, config.projection_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class TerraVisionEncoder(nn.Module):
    """Complete vision encoder: patches → ViT → projection → LLM-compatible embeddings.

    Output shape: (batch, num_image_tokens, projection_size)
    These embeddings can be directly concatenated with text token embeddings.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projection = VisionProjection(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) normalized images
        Returns:
            (B, num_patches+1, projection_size) embeddings ready for LLM
        """
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Project all tokens (CLS + patches) into LLM space
        return self.projection(x)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_millions": total / 1e6}

    def save_pretrained(self, path: str):
        import json
        from pathlib import Path as P
        from safetensors.torch import save_file

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(save_dir / "vision_encoder.safetensors"))
        (save_dir / "vision_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TerraVisionEncoder":
        import json
        from pathlib import Path as P
        from safetensors.torch import load_file

        config = VisionConfig.from_dict(json.loads((P(path) / "vision_config.json").read_text()))
        model = cls(config)
        state_dict = load_file(str(P(path) / "vision_encoder.safetensors"), device=device)
        model.load_state_dict(state_dict)
        return model
