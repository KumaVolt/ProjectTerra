"""Audio encoder for Terra multimodal model (Speech-to-Text).

A Whisper-style encoder that converts mel spectrograms into embeddings
compatible with Terra's text transformer. Can be trained independently.

Architecture:
- Conv1d stem (mel → features)
- Sinusoidal position encoding
- Transformer encoder blocks
- MLP projection → Terra hidden_size

Training approach:
1. Pre-train with CTC loss on transcription (ASR task)
2. Fine-tune the projection to align with Terra's embedding space
3. Joint fine-tune on speech-text instruction data
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioConfig:
    """Audio encoder configuration."""

    # Mel spectrogram input
    num_mel_bins: int = 80
    max_audio_length: int = 3000  # ~30 seconds at 100 frames/sec

    # Encoder
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    layer_norm_eps: float = 1e-5
    dropout: float = 0.1

    # Projection to LLM space
    projection_size: int = 896  # Must match TerraConfig.hidden_size

    # CTC head (for standalone ASR pre-training)
    ctc_vocab_size: int = 32000  # Match Terra tokenizer

    @classmethod
    def audio_tiny(cls, projection_size: int = 896) -> "AudioConfig":
        """~5M params - fast experiments."""
        return cls(
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            projection_size=projection_size,
        )

    @classmethod
    def audio_small(cls, projection_size: int = 896) -> "AudioConfig":
        """~20M params - good for consumer hardware."""
        return cls(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=6,
            num_attention_heads=4,
            projection_size=projection_size,
        )

    @classmethod
    def audio_base(cls, projection_size: int = 896) -> "AudioConfig":
        """~60M params - high quality."""
        return cls(
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=8,
            num_attention_heads=8,
            projection_size=projection_size,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def param_count_estimate(self) -> int:
        conv_stem = self.num_mel_bins * self.hidden_size * 3 + self.hidden_size * self.hidden_size * 3
        attn = 4 * self.hidden_size * self.hidden_size
        ffn = 2 * self.hidden_size * self.intermediate_size
        per_layer = attn + ffn + 2 * self.hidden_size
        projection = self.hidden_size * self.projection_size * 2
        return conv_stem + self.num_hidden_layers * per_layer + projection


class AudioConvStem(nn.Module):
    """Conv1d stem to downsample mel spectrogram and extract initial features.

    Two conv layers with stride 2 each → 4x temporal downsampling.
    30 seconds of audio at 100fps → 750 frames → 187 tokens (manageable for attention).
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, num_mel_bins, time_steps)
        x = self.gelu(self.conv1(mel))
        x = self.gelu(self.conv2(x))
        # (B, hidden_size, time_steps/4) → (B, time_steps/4, hidden_size)
        return x.transpose(1, 2)


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (Whisper-style)."""

    def __init__(self, hidden_size: int, max_len: int = 8000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]


class AudioAttention(nn.Module):
    def __init__(self, config: AudioConfig):
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


class AudioMLP(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class AudioBlock(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = AudioAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = AudioMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AudioProjection(nn.Module):
    """Project audio features → LLM embedding space."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.projection_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(config.projection_size, config.projection_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class TerraAudioEncoder(nn.Module):
    """Complete audio encoder: mel spectrogram → features → projection → LLM embeddings.

    Output: (batch, num_audio_tokens, projection_size)
    These embeddings can be directly concatenated with text token embeddings.

    Two modes:
    1. ASR pre-training: use ctc_head for CTC loss (standalone, no LLM needed)
    2. Multimodal: use projection to feed into Terra LLM
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.conv_stem = AudioConvStem(config)
        self.pos_enc = SinusoidalPositionEncoding(config.hidden_size)
        self.blocks = nn.ModuleList([AudioBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projection = AudioProjection(config)
        # CTC head for standalone ASR pre-training
        self.ctc_head = nn.Linear(config.hidden_size, config.ctc_vocab_size)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to hidden features (before projection).

        Args:
            mel: (B, num_mel_bins, time_steps)
        Returns:
            (B, num_frames, hidden_size)
        """
        x = self.conv_stem(mel)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Full forward: mel → projected embeddings for LLM.

        Args:
            mel: (B, num_mel_bins, time_steps)
        Returns:
            (B, num_audio_tokens, projection_size) - ready for Terra LLM
        """
        features = self.encode(mel)
        return self.projection(features)

    def forward_ctc(self, mel: torch.Tensor) -> torch.Tensor:
        """CTC forward for standalone ASR pre-training.

        Args:
            mel: (B, num_mel_bins, time_steps)
        Returns:
            (B, num_frames, ctc_vocab_size) log-probabilities
        """
        features = self.encode(mel)
        return F.log_softmax(self.ctc_head(features), dim=-1)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_millions": total / 1e6}

    def save_pretrained(self, path: str):
        import json
        from pathlib import Path as P
        from safetensors.torch import save_file

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(save_dir / "audio_encoder.safetensors"))
        (save_dir / "audio_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TerraAudioEncoder":
        import json
        from pathlib import Path as P
        from safetensors.torch import load_file

        config = AudioConfig.from_dict(json.loads((P(path) / "audio_config.json").read_text()))
        model = cls(config)
        state_dict = load_file(str(P(path) / "audio_encoder.safetensors"), device=device)
        model.load_state_dict(state_dict)
        return model
