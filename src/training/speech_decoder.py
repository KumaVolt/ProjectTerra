"""Speech decoder (Text-to-Speech) for Terra multimodal model.

Converts Terra's hidden states into audio waveforms. Uses a neural codec approach:
the LLM generates discrete audio tokens, and a lightweight vocoder converts them to audio.

Architecture:
1. Audio Tokenizer: encodes/decodes audio ↔ discrete tokens (like EnCodec/SpeechTokenizer)
2. LLM integration: Terra generates audio tokens interleaved with text tokens
3. Vocoder: converts audio tokens → waveform

For training:
1. Pre-train the codec (encoder + decoder + quantizer) on speech data
2. Add audio token vocabulary to Terra's tokenizer
3. Fine-tune Terra to generate audio tokens conditioned on text

For full-duplex: the decoder must support streaming (generate audio chunk by chunk).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeechDecoderConfig:
    """Speech decoder (vocoder) configuration."""

    # Audio codec
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    num_mel_bins: int = 80

    # Codec encoder/decoder
    hidden_size: int = 256
    num_residual_layers: int = 3
    upsample_rates: tuple[int, ...] = (8, 5, 4, 2)  # Total: 320x upsampling

    # Vector quantization
    codebook_size: int = 1024
    num_codebooks: int = 4  # Residual VQ layers
    codebook_dim: int = 128

    # Conditioning from LLM
    lm_hidden_size: int = 896  # Terra's hidden_size

    dropout: float = 0.1

    @classmethod
    def speech_tiny(cls, lm_hidden_size: int = 896) -> "SpeechDecoderConfig":
        """~3M params - fast experiments."""
        return cls(
            hidden_size=128,
            num_residual_layers=2,
            upsample_rates=(8, 5, 4, 2),
            codebook_size=512,
            num_codebooks=2,
            codebook_dim=64,
            lm_hidden_size=lm_hidden_size,
        )

    @classmethod
    def speech_small(cls, lm_hidden_size: int = 896) -> "SpeechDecoderConfig":
        """~12M params - good quality."""
        return cls(
            hidden_size=256,
            num_residual_layers=3,
            upsample_rates=(8, 5, 4, 2),
            codebook_size=1024,
            num_codebooks=4,
            codebook_dim=128,
            lm_hidden_size=lm_hidden_size,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SpeechDecoderConfig":
        valid = {}
        for k, v in d.items():
            if k in cls.__dataclass_fields__:
                if k == "upsample_rates" and isinstance(v, list):
                    v = tuple(v)
                valid[k] = v
        return cls(**valid)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["upsample_rates"] = list(d["upsample_rates"])
        return d

    def param_count_estimate(self) -> int:
        # Rough estimate
        encoder = self.num_mel_bins * self.hidden_size * 3 * self.num_residual_layers
        decoder = self.hidden_size * self.hidden_size * 3 * self.num_residual_layers * len(self.upsample_rates)
        codebook = self.num_codebooks * self.codebook_size * self.codebook_dim
        conditioning = self.lm_hidden_size * self.hidden_size
        return encoder + decoder + codebook + conditioning


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.silu(self.norm1(self.conv1(x))))
        x = self.dropout(self.norm2(self.conv2(x)))
        return x + residual


class VectorQuantizer(nn.Module):
    """Single codebook vector quantizer with EMA updates."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, codebook_dim, T) continuous features
        Returns:
            quantized: (B, codebook_dim, T)
            indices: (B, T) codebook indices
            commit_loss: commitment loss scalar
        """
        # (B, D, T) → (B*T, D)
        z_flat = z.permute(0, 2, 1).reshape(-1, self.codebook_dim)

        # Find nearest codebook entries
        distances = torch.cdist(z_flat, self.embedding.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.embedding(indices)

        # Reshape back
        B, D, T = z.shape
        quantized = quantized.view(B, T, D).permute(0, 2, 1)
        indices = indices.view(B, T)

        # Commitment loss
        commit_loss = F.mse_loss(z, quantized.detach())

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, indices, commit_loss


class ResidualVQ(nn.Module):
    """Residual Vector Quantization — multiple VQ layers for better reconstruction."""

    def __init__(self, config: SpeechDecoderConfig):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.pre_proj = nn.Conv1d(config.hidden_size, config.codebook_dim, 1)
        self.post_proj = nn.Conv1d(config.codebook_dim, config.hidden_size, 1)
        self.quantizers = nn.ModuleList([
            VectorQuantizer(config.codebook_size, config.codebook_dim)
            for _ in range(config.num_codebooks)
        ])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Args:
            z: (B, hidden_size, T)
        Returns:
            quantized: (B, hidden_size, T)
            all_indices: list of (B, T) per codebook
            total_loss: summed commitment loss
        """
        z_proj = self.pre_proj(z)
        residual = z_proj
        quantized_sum = torch.zeros_like(z_proj)
        all_indices = []
        total_loss = torch.tensor(0.0, device=z.device)

        for vq in self.quantizers:
            quantized, indices, loss = vq(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            all_indices.append(indices)
            total_loss = total_loss + loss

        return self.post_proj(quantized_sum), all_indices, total_loss

    def decode_tokens(self, token_ids: list[torch.Tensor]) -> torch.Tensor:
        """Convert codebook indices back to features.

        Args:
            token_ids: list of (B, T) tensors, one per codebook
        Returns:
            (B, hidden_size, T)
        """
        quantized_sum = torch.zeros(
            token_ids[0].shape[0],
            self.quantizers[0].codebook_dim,
            token_ids[0].shape[1],
            device=token_ids[0].device,
        )
        for vq, ids in zip(self.quantizers, token_ids):
            emb = vq.embedding(ids)  # (B, T, D)
            quantized_sum = quantized_sum + emb.permute(0, 2, 1)
        return self.post_proj(quantized_sum)


class AudioCodecEncoder(nn.Module):
    """Encode raw audio / mel spectrogram → latent features for VQ."""

    def __init__(self, config: SpeechDecoderConfig):
        super().__init__()
        self.conv_in = nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            ResidualBlock(config.hidden_size, dilation=3**i, dropout=config.dropout)
            for i in range(config.num_residual_layers)
        ])
        # Downsample to match VQ frame rate
        self.downsample = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=4, stride=2, padding=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, num_mel_bins, T)
        Returns:
            (B, hidden_size, T')
        """
        x = self.conv_in(mel)
        for block in self.blocks:
            x = block(x)
        return self.downsample(x)


class AudioCodecDecoder(nn.Module):
    """Decode VQ features → mel spectrogram / waveform."""

    def __init__(self, config: SpeechDecoderConfig):
        super().__init__()
        # Upsample from VQ frame rate
        self.upsample = nn.ConvTranspose1d(config.hidden_size, config.hidden_size, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock(config.hidden_size, dilation=3**i, dropout=config.dropout)
            for i in range(config.num_residual_layers)
        ])
        self.conv_out = nn.Conv1d(config.hidden_size, config.num_mel_bins, kernel_size=7, padding=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, hidden_size, T')
        Returns:
            (B, num_mel_bins, T) reconstructed mel
        """
        x = self.upsample(z)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)


class LMConditioner(nn.Module):
    """Conditions the audio decoder on Terra LLM hidden states.

    Takes LLM hidden states and projects them to match audio feature rate/dim.
    """

    def __init__(self, config: SpeechDecoderConfig):
        super().__init__()
        self.proj = nn.Linear(config.lm_hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, lm_hidden: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            lm_hidden: (B, lm_seq, lm_hidden_size) from Terra LLM
            target_len: target temporal length for audio features
        Returns:
            (B, hidden_size, target_len) conditioning signal
        """
        x = self.norm(self.proj(lm_hidden))  # (B, lm_seq, hidden_size)
        x = x.permute(0, 2, 1)  # (B, hidden_size, lm_seq)
        # Interpolate to match audio frame rate
        if x.shape[2] != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x


class TerraSpeechDecoder(nn.Module):
    """Complete speech decoder: LLM states + audio tokens → reconstructed audio.

    Training pipeline:
    1. Pre-train codec (encoder + VQ + decoder) on speech reconstruction
    2. Freeze codec, train LM conditioner to generate audio tokens from LLM states
    3. Joint fine-tune on speech generation tasks

    Inference:
    - Terra LLM generates audio token IDs → VQ.decode_tokens → decoder → mel → vocoder → wav
    """

    def __init__(self, config: SpeechDecoderConfig):
        super().__init__()
        self.config = config
        self.codec_encoder = AudioCodecEncoder(config)
        self.residual_vq = ResidualVQ(config)
        self.codec_decoder = AudioCodecDecoder(config)
        self.lm_conditioner = LMConditioner(config)

    def encode_audio(self, mel: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encode mel spectrogram to discrete audio tokens.

        Args:
            mel: (B, num_mel_bins, T)
        Returns:
            token_ids: list of (B, T') per codebook — these get added to LLM vocab
            commit_loss: VQ commitment loss
        """
        z = self.codec_encoder(mel)
        _, token_ids, commit_loss = self.residual_vq(z)
        return token_ids, commit_loss

    def decode_tokens(self, token_ids: list[torch.Tensor]) -> torch.Tensor:
        """Decode audio token IDs → mel spectrogram.

        Args:
            token_ids: list of (B, T') per codebook
        Returns:
            (B, num_mel_bins, T) reconstructed mel
        """
        z = self.residual_vq.decode_tokens(token_ids)
        return self.codec_decoder(z)

    def forward_codec(self, mel: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Full codec forward: mel → encode → VQ → decode → reconstructed mel.

        Used for pre-training the codec on reconstruction loss.
        """
        z = self.codec_encoder(mel)
        z_q, token_ids, commit_loss = self.residual_vq(z)
        mel_hat = self.codec_decoder(z_q)
        return mel_hat, token_ids, commit_loss

    def forward_conditioned(
        self, lm_hidden: torch.Tensor, target_token_ids: list[torch.Tensor]
    ) -> torch.Tensor:
        """Generate mel conditioned on LLM hidden states.

        Used during fine-tuning when Terra LLM provides context.

        Args:
            lm_hidden: (B, lm_seq, lm_hidden_size) from Terra
            target_token_ids: ground-truth audio tokens for teacher forcing
        Returns:
            (B, num_mel_bins, T) predicted mel
        """
        z_target = self.residual_vq.decode_tokens(target_token_ids)
        target_len = z_target.shape[2]
        conditioning = self.lm_conditioner(lm_hidden, target_len)
        # Add conditioning to decoded features
        z_conditioned = z_target + conditioning
        return self.codec_decoder(z_conditioned)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_millions": total / 1e6}

    def save_pretrained(self, path: str):
        import json
        from pathlib import Path as P
        from safetensors.torch import save_file

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(save_dir / "speech_decoder.safetensors"))
        (save_dir / "speech_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TerraSpeechDecoder":
        import json
        from pathlib import Path as P
        from safetensors.torch import load_file

        config = SpeechDecoderConfig.from_dict(json.loads((P(path) / "speech_config.json").read_text()))
        model = cls(config)
        state_dict = load_file(str(P(path) / "speech_decoder.safetensors"), device=device)
        model.load_state_dict(state_dict)
        return model
