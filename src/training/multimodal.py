"""Terra Unified Multimodal Model — Moshi/PersonaPlex-inspired architecture.

Architecture (adapted for small models ~150M):

    multimodal tokenizers (VQ-VAE for images, Mimi-style codec for speech)
                    ↓
    temporal transformer (main model, processes 1 token per audio timestep)
                    ↓
    depth transformer (small, expands semantic token → all 8 codebook tokens)
                    ↓
    modality decoders (text head, VQ-VAE decoder, codec decoder)

Key design from Moshi/PersonaPlex adapted for small model:

1. Audio codec with 8 codebooks (like Mimi). Codebook 0 = semantic (meaning),
   codebooks 1-7 = acoustic (sound quality). Only semantic tokens go into the
   main transformer → much shorter sequences.

2. Depth Transformer: tiny model (~2M params) that autoregressively predicts
   codebooks 1-7 given the temporal transformer's hidden state. Runs per-timestep.

3. Dual-stream for full-duplex: user audio + agent audio as parallel token streams.
   The temporal transformer sees interleaved [user_semantic_t0, agent_semantic_t0, ...]

Token layout in the temporal transformer vocabulary:
    [0, text_vocab)                              → text tokens
    [text_vocab, text_vocab + img_codebook)       → image VQ tokens
    [text_vocab + img_codebook, ... + audio_cb)   → audio semantic tokens (codebook 0)

Depth transformer has its own vocab: 7 codebooks x codebook_size.

Special tokens:
    <|image_start|> <|image_end|>
    <|audio_start|> <|audio_end|>
    <|generate_image|> <|generate_audio|>
    <|user_audio|> <|agent_audio|>
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.model import TerraConfig, TerraForCausalLM


# ── Special Token IDs ──

# Must match indices in src/training/tokenizer.py SPECIAL_TOKENS list
SPECIAL_TOKENS = {
    "image_start": 6,        # <|image|>
    "image_end": 7,          # <|/image|>
    "audio_start": 8,        # <|audio|>
    "audio_end": 9,          # <|/audio|>
    "generate_image": 6,     # reuse image_start as trigger
    "generate_audio": 8,     # reuse audio_start as trigger
    "user_audio": 10,        # <|user_audio|>
    "agent_audio": 11,       # <|agent_audio|>
}


# ── Configuration ──

@dataclass
class UnifiedMultimodalConfig:
    """Configuration for the unified multimodal model."""

    # Text backbone
    text_vocab_size: int = 32000

    # Image tokenizer (VQ-VAE)
    image_size: int = 256
    image_patch_size: int = 16
    image_codebook_size: int = 8192
    image_codebook_dim: int = 256
    image_encoder_hidden: int = 256
    image_encoder_layers: int = 4
    image_num_tokens: int = 256  # (image_size/patch_size)^2

    # Audio tokenizer (Mimi-style codec)
    audio_codebook_size: int = 1024
    audio_num_codebooks: int = 8       # 8 codebooks like Mimi (was 4)
    audio_codebook_dim: int = 128
    audio_encoder_hidden: int = 256
    audio_encoder_layers: int = 4
    num_mel_bins: int = 80

    # Depth transformer (predicts codebooks 1-7 from temporal hidden state)
    depth_hidden_size: int = 256
    depth_num_layers: int = 2
    depth_num_heads: int = 4
    depth_dropout: float = 0.1

    # Freeze backbone when training tokenizers
    freeze_text_backbone: bool = True

    @property
    def temporal_vocab_size(self) -> int:
        """Vocab for the main (temporal) transformer.
        Only includes text + image + audio semantic (codebook 0).
        """
        return self.text_vocab_size + self.image_codebook_size + self.audio_codebook_size

    @property
    def depth_vocab_size(self) -> int:
        """Vocab for the depth transformer: codebooks 1 through num_codebooks-1."""
        return self.audio_codebook_size * (self.audio_num_codebooks - 1)

    @property
    def image_token_offset(self) -> int:
        return self.text_vocab_size

    @property
    def audio_semantic_offset(self) -> int:
        """Offset for semantic audio tokens (codebook 0) in temporal vocab."""
        return self.text_vocab_size + self.image_codebook_size

    def depth_codebook_offset(self, codebook_idx: int) -> int:
        """Offset for codebook in depth transformer vocab (1-indexed, cb1=0)."""
        return (codebook_idx - 1) * self.audio_codebook_size

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedMultimodalConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ── Image Tokenizer (VQ-VAE) — unchanged ──

class VectorQuantizer(nn.Module):
    """Single codebook vector quantizer with straight-through estimator."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D = z.shape[0], z.shape[1]
        spatial = z.shape[2:]
        z_flat = z.reshape(B, D, -1).permute(0, 2, 1).reshape(-1, D)

        distances = torch.cdist(z_flat, self.embedding.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.embedding(indices)

        quantized = quantized.view(B, -1, D).permute(0, 2, 1).view(B, D, *spatial)
        indices = indices.view(B, -1)

        commit_loss = F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach()

        return quantized, indices, commit_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(indices)
        return emb.permute(0, 2, 1)


class ImageVQEncoder(nn.Module):
    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        h = config.image_encoder_hidden
        d = config.image_codebook_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, h // 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(h // 2, h, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(h, h, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(h, d, 4, stride=2, padding=1), nn.SiLU(),
        )
        self.vq = VectorQuantizer(config.image_codebook_size, d)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(images)
        return self.vq(z)

    def encode_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        z = self.encoder(images)
        _, token_ids, _ = self.vq(z)
        return token_ids


class ImageVQDecoder(nn.Module):
    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        h = config.image_encoder_hidden
        d = config.image_codebook_dim
        self.spatial_size = config.image_size // 16
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d, h, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(h, h, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(h, h // 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(h // 2, 3, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)

    def decode_tokens(self, token_ids: torch.Tensor, vq: VectorQuantizer) -> torch.Tensor:
        emb = vq.decode_indices(token_ids)
        B, D, N = emb.shape
        h = w = int(math.sqrt(N))
        return self.decoder(emb.view(B, D, h, w))


class ImageTokenizer(nn.Module):
    """Image VQ-VAE tokenizer: images <-> discrete tokens."""

    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        self.config = config
        self.encoder = ImageVQEncoder(config)
        self.decoder = ImageVQDecoder(config)
        self.vq = self.encoder.vq

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized, token_ids, commit_loss = self.encoder(images)
        reconstructed = self.decoder(quantized)
        recon_loss = F.mse_loss(reconstructed, images)
        return reconstructed, token_ids, recon_loss + 0.25 * commit_loss

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_to_tokens(images) + self.config.image_token_offset

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode_tokens(token_ids - self.config.image_token_offset, self.vq)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_millions": total / 1e6}

    def save_pretrained(self, path: str):
        import json
        from pathlib import Path as P
        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(save_dir / "image_tokenizer.pt"))
        (save_dir / "image_tokenizer_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "ImageTokenizer":
        import json
        from pathlib import Path as P
        config = UnifiedMultimodalConfig.from_dict(json.loads((P(path) / "image_tokenizer_config.json").read_text()))
        model = cls(config)
        state = torch.load(str(P(path) / "image_tokenizer.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model


# ── Audio Tokenizer (Mimi-style Codec) ──

class AudioCodecEncoder(nn.Module):
    """Encode mel spectrogram -> latent features for RVQ."""

    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        h = config.audio_encoder_hidden
        self.conv_in = nn.Conv1d(config.num_mel_bins, h, 7, padding=3)
        layers = []
        for i in range(config.audio_encoder_layers):
            layers.extend([
                nn.Conv1d(h, h, 3, dilation=3**i, padding=3**i),
                nn.SiLU(),
                nn.GroupNorm(1, h),
            ])
        self.blocks = nn.Sequential(*layers)
        self.downsample = nn.Conv1d(h, config.audio_codebook_dim, 4, stride=2, padding=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.conv_in(mel))
        x = self.blocks(x)
        return self.downsample(x)


class AudioCodecDecoder(nn.Module):
    """Decode RVQ features -> mel spectrogram."""

    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        h = config.audio_encoder_hidden
        self.upsample = nn.ConvTranspose1d(config.audio_codebook_dim, h, 4, stride=2, padding=1)
        layers = []
        for i in range(config.audio_encoder_layers):
            layers.extend([
                nn.Conv1d(h, h, 3, dilation=3**i, padding=3**i),
                nn.SiLU(),
                nn.GroupNorm(1, h),
            ])
        self.blocks = nn.Sequential(*layers)
        self.conv_out = nn.Conv1d(h, config.num_mel_bins, 7, padding=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.upsample(z))
        x = self.blocks(x)
        return self.conv_out(x)


class ResidualVQ(nn.Module):
    """Residual Vector Quantization with 8 codebooks (Mimi-style).

    Codebook 0 captures semantic content (meaning).
    Codebooks 1-7 capture acoustic detail (sound quality).
    """

    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        self.num_codebooks = config.audio_num_codebooks
        self.quantizers = nn.ModuleList([
            VectorQuantizer(config.audio_codebook_size, config.audio_codebook_dim)
            for _ in range(config.audio_num_codebooks)
        ])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        total_loss = torch.tensor(0.0, device=z.device)

        for vq in self.quantizers:
            quantized, indices, loss = vq(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            all_indices.append(indices)
            total_loss = total_loss + loss

        return quantized_sum, all_indices, total_loss

    def decode_tokens(self, token_ids_per_codebook: list[torch.Tensor]) -> torch.Tensor:
        quantized_sum = None
        for vq, ids in zip(self.quantizers, token_ids_per_codebook):
            emb = vq.decode_indices(ids)
            quantized_sum = emb if quantized_sum is None else quantized_sum + emb
        return quantized_sum


class AudioTokenizer(nn.Module):
    """Mimi-style audio codec: mel <-> 8 codebooks of discrete tokens.

    Unlike the previous flat interleaving, this codec separates:
    - Semantic tokens (codebook 0): go into the temporal transformer
    - Acoustic tokens (codebooks 1-7): predicted by the depth transformer

    The temporal transformer only sees 1 token per audio timestep, not 8.
    """

    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioCodecEncoder(config)
        self.rvq = ResidualVQ(config)
        self.decoder = AudioCodecDecoder(config)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Full codec: mel -> encode -> RVQ -> decode -> reconstructed mel.
        Returns: (reconstructed_mel, token_ids_per_codebook, loss)
        """
        z = self.encoder(mel)
        z_q, token_ids_list, commit_loss = self.rvq(z)
        mel_hat = self.decoder(z_q)
        recon_loss = F.l1_loss(mel_hat, mel)
        return mel_hat, token_ids_list, recon_loss + 0.25 * commit_loss

    def encode(self, mel: torch.Tensor) -> list[torch.Tensor]:
        """Mel -> list of token IDs per codebook. Each is (B, T')."""
        z = self.encoder(mel)
        _, token_ids_list, _ = self.rvq(z)
        return token_ids_list

    def encode_semantic(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel -> semantic tokens only (codebook 0), offset for temporal vocab.
        (B, mel_bins, T) -> (B, T')
        """
        token_ids_list = self.encode(mel)
        return token_ids_list[0] + self.config.audio_semantic_offset

    def decode_from_all_codebooks(self, token_ids_list: list[torch.Tensor]) -> torch.Tensor:
        """All 8 codebook token IDs -> mel spectrogram."""
        z_q = self.rvq.decode_tokens(token_ids_list)
        return self.decoder(z_q)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_millions": total / 1e6}

    def save_pretrained(self, path: str):
        import json
        from pathlib import Path as P
        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(save_dir / "audio_tokenizer.pt"))
        (save_dir / "audio_tokenizer_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "AudioTokenizer":
        import json
        from pathlib import Path as P
        config = UnifiedMultimodalConfig.from_dict(json.loads((P(path) / "audio_tokenizer_config.json").read_text()))
        model = cls(config)
        state = torch.load(str(P(path) / "audio_tokenizer.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model


# ── Depth Transformer (Moshi-style) ──

class DepthTransformerBlock(nn.Module):
    """Single transformer block for the depth transformer."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        h = self.norm1(x)
        if is_causal:
            seq_len = h.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=h.device)
            h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        else:
            h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class DepthTransformer(nn.Module):
    """Small transformer that predicts acoustic codebooks 1-7 from the
    temporal transformer's hidden state at each audio timestep.

    Moshi insight: the big temporal transformer processes the sequence with
    1 token per audio frame (semantic codebook 0). Then this small depth
    transformer expands each frame into all 8 codebook tokens.

    This is much more efficient than putting all 8 codebooks into the
    main sequence (which would 8x the sequence length).

    For a ~150M main model, this adds only ~2M params.
    """

    def __init__(self, config: UnifiedMultimodalConfig):
        super().__init__()
        self.config = config
        h = config.depth_hidden_size
        num_acoustic = config.audio_num_codebooks - 1  # codebooks 1-7

        # Project temporal transformer hidden state -> depth hidden
        self.temporal_proj = nn.Linear(config.text_vocab_size, h)  # placeholder, set in TerraMultimodal

        # Embeddings for acoustic codebook tokens
        self.token_embeddings = nn.Embedding(config.audio_codebook_size * num_acoustic, h)

        # Codebook position embedding (which codebook are we predicting?)
        self.codebook_pos = nn.Embedding(num_acoustic, h)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DepthTransformerBlock(h, config.depth_num_heads, config.depth_dropout)
            for _ in range(config.depth_num_layers)
        ])
        self.norm = nn.LayerNorm(h)

        # Output head: predict next codebook token
        self.head = nn.Linear(h, config.audio_codebook_size)

    def set_temporal_dim(self, temporal_hidden_size: int):
        """Set the projection from temporal transformer hidden size."""
        self.temporal_proj = nn.Linear(temporal_hidden_size, self.config.depth_hidden_size)

    def forward(
        self,
        temporal_hidden: torch.Tensor,
        target_codebook_tokens: list[torch.Tensor] | None = None,
    ) -> dict:
        """Predict acoustic codebook tokens for one audio timestep.

        Args:
            temporal_hidden: (B, hidden) from temporal transformer at this timestep
            target_codebook_tokens: list of (B,) for codebooks 1-7 (for training)
        Returns:
            dict with 'logits' list and optional 'loss'
        """
        B = temporal_hidden.shape[0]
        h = self.config.depth_hidden_size
        num_acoustic = self.config.audio_num_codebooks - 1
        device = temporal_hidden.device

        # Start with temporal hidden state as first "token"
        temporal_emb = self.temporal_proj(temporal_hidden).unsqueeze(1)  # (B, 1, h)

        if target_codebook_tokens is not None:
            # Training: teacher forcing — build full sequence and compute loss
            seq = [temporal_emb]
            for i in range(num_acoustic - 1):  # don't need last target as input
                cb_offset = self.config.depth_codebook_offset(i + 1)
                tok_emb = self.token_embeddings(target_codebook_tokens[i] + cb_offset)
                pos_emb = self.codebook_pos(torch.tensor(i, device=device))
                seq.append((tok_emb + pos_emb).unsqueeze(1))

            x = torch.cat(seq, dim=1)  # (B, num_acoustic, h)
            for block in self.blocks:
                x = block(x, is_causal=True)
            x = self.norm(x)

            all_logits = self.head(x)  # (B, num_acoustic, codebook_size)
            # Loss: predict each codebook from previous ones
            loss = torch.tensor(0.0, device=device)
            for i in range(num_acoustic):
                loss = loss + F.cross_entropy(all_logits[:, i], target_codebook_tokens[i])
            loss = loss / num_acoustic

            return {"logits": [all_logits[:, i] for i in range(num_acoustic)], "loss": loss}
        else:
            # Inference: autoregressive generation
            all_logits = []
            x = temporal_emb  # (B, 1, h)

            for i in range(num_acoustic):
                for block in self.blocks:
                    x = block(x, is_causal=True)
                out = self.norm(x)
                logits_i = self.head(out[:, -1])  # (B, codebook_size)
                all_logits.append(logits_i)

                # Sample next token and append
                next_token = logits_i.argmax(dim=-1)  # greedy for acoustic
                cb_offset = self.config.depth_codebook_offset(i + 1)
                tok_emb = self.token_embeddings(next_token + cb_offset)
                pos_emb = self.codebook_pos(torch.tensor(i, device=device))
                next_emb = (tok_emb + pos_emb).unsqueeze(1)
                x = torch.cat([x, next_emb], dim=1)

            return {"logits": all_logits}

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_millions": total / 1e6}


# ── Unified Multimodal Model ──

class TerraMultimodal(nn.Module):
    """Unified multimodal model with Moshi-style depth transformer.

    Temporal transformer (main, ~150M):
        Processes text + image tokens + audio SEMANTIC tokens (1 per timestep).
        Sequence: "describe this" <|img_start|> [256 img tokens] <|img_end|> ...
        For audio: ... <|audio_start|> [semantic_t0] [semantic_t1] ... <|audio_end|>

    Depth transformer (tiny, ~2M):
        For each audio timestep, takes the temporal hidden state and predicts
        codebooks 1-7 (acoustic detail). Runs independently per timestep.

    Full-duplex (PersonaPlex-style):
        Two audio streams interleaved in the temporal sequence:
        <|audio_start|> <|user_audio|> [sem_t0] <|agent_audio|> [sem_t0]
                        <|user_audio|> [sem_t1] <|agent_audio|> [sem_t1] ...
        The model listens to user semantic tokens while generating agent tokens.
        Depth transformer only expands AGENT audio to full codebooks for synthesis.
    """

    def __init__(
        self,
        text_model: TerraForCausalLM,
        config: UnifiedMultimodalConfig,
        image_tokenizer: ImageTokenizer | None = None,
        audio_tokenizer: AudioTokenizer | None = None,
    ):
        super().__init__()
        self.config = config
        self.text_model = text_model

        # Expand temporal transformer vocab: text + image + audio semantic
        old_vocab = text_model.config.vocab_size
        new_vocab = config.temporal_vocab_size
        if new_vocab > old_vocab:
            self._expand_embeddings(old_vocab, new_vocab)

        self.image_tokenizer = image_tokenizer
        self.audio_tokenizer = audio_tokenizer

        # Depth transformer for acoustic codebooks
        self.depth_transformer = DepthTransformer(config)
        self.depth_transformer.set_temporal_dim(text_model.config.hidden_size)

        if config.freeze_text_backbone:
            for param in self.text_model.named_parameters():
                param[1].requires_grad = False
            # Unfreeze expanded embeddings
            self.text_model.model.embed_tokens.weight.requires_grad = True
            if self.text_model.lm_head is not None:
                self.text_model.lm_head.weight.requires_grad = True

    def _expand_embeddings(self, old_vocab: int, new_vocab: int):
        old_emb = self.text_model.model.embed_tokens
        new_emb = nn.Embedding(new_vocab, old_emb.embedding_dim)
        with torch.no_grad():
            new_emb.weight[:old_vocab] = old_emb.weight
            nn.init.normal_(new_emb.weight[old_vocab:], mean=0.0, std=0.02)
        self.text_model.model.embed_tokens = new_emb
        self.text_model.config.vocab_size = new_vocab

        if self.text_model.lm_head is not None:
            old_head = self.text_model.lm_head
            new_head = nn.Linear(old_head.in_features, new_vocab, bias=False)
            with torch.no_grad():
                new_head.weight[:old_vocab] = old_head.weight
                nn.init.normal_(new_head.weight[old_vocab:], mean=0.0, std=0.02)
            self.text_model.lm_head = new_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass through temporal transformer.

        Input contains text + image + semantic audio tokens.
        The temporal transformer sees only 1 audio token per timestep (not 8).
        """
        return self.text_model(input_ids, attention_mask, labels)

    def forward_with_depth(
        self,
        input_ids: torch.Tensor,
        audio_positions: torch.Tensor,
        target_acoustic_tokens: list[torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward through temporal + depth transformer.

        Args:
            input_ids: (B, seq) temporal token sequence
            audio_positions: (B, num_audio_frames) indices into seq where audio tokens are
            target_acoustic_tokens: list of 7 tensors (B, num_audio_frames) for training
            labels: (B, seq) for temporal transformer loss
        Returns:
            dict with temporal_loss, depth_loss, total_loss
        """
        # Temporal forward
        hidden_states = self.text_model.model(input_ids)
        logits = F.linear(hidden_states, self.text_model.get_lm_head_weight())

        temporal_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            temporal_loss = F.cross_entropy(
                shift_logits.view(-1, self.text_model.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Depth forward at audio positions
        depth_loss = None
        if audio_positions is not None and audio_positions.shape[1] > 0:
            B = hidden_states.shape[0]
            # Gather hidden states at audio timestep positions
            audio_hidden = torch.gather(
                hidden_states,
                1,
                audio_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
            )  # (B, num_frames, hidden)

            # Run depth transformer for each audio frame
            depth_losses = []
            for t in range(audio_positions.shape[1]):
                frame_hidden = audio_hidden[:, t]  # (B, hidden)
                if target_acoustic_tokens is not None:
                    frame_targets = [cb[:, t] for cb in target_acoustic_tokens]
                    result = self.depth_transformer(frame_hidden, frame_targets)
                    depth_losses.append(result["loss"])
                else:
                    self.depth_transformer(frame_hidden)

            if depth_losses:
                depth_loss = torch.stack(depth_losses).mean()

        total_loss = None
        if temporal_loss is not None:
            total_loss = temporal_loss
            if depth_loss is not None:
                total_loss = total_loss + depth_loss

        return {
            "logits": logits,
            "temporal_loss": temporal_loss,
            "depth_loss": depth_loss,
            "loss": total_loss,
        }

    # ── Encoding helpers ──

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_tokenizer is None:
            raise RuntimeError("No image tokenizer")
        return self.image_tokenizer.encode(images)

    def decode_image(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.image_tokenizer is None:
            raise RuntimeError("No image tokenizer")
        return self.image_tokenizer.decode(token_ids)

    def encode_audio_semantic(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode audio to semantic tokens for temporal transformer."""
        if self.audio_tokenizer is None:
            raise RuntimeError("No audio tokenizer")
        return self.audio_tokenizer.encode_semantic(mel)

    def encode_audio_all(self, mel: torch.Tensor) -> list[torch.Tensor]:
        """Encode audio to all 8 codebook token lists."""
        if self.audio_tokenizer is None:
            raise RuntimeError("No audio tokenizer")
        return self.audio_tokenizer.encode(mel)

    def decode_audio(self, token_ids_list: list[torch.Tensor]) -> torch.Tensor:
        """Decode all 8 codebooks back to mel."""
        if self.audio_tokenizer is None:
            raise RuntimeError("No audio tokenizer")
        return self.audio_tokenizer.decode_from_all_codebooks(token_ids_list)

    # ── Multimodal input building ──

    def build_multimodal_input(
        self,
        text_tokens: torch.Tensor,
        images: torch.Tensor | None = None,
        mel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build temporal token sequence from text + optional images + optional audio.

        Audio only uses SEMANTIC tokens (codebook 0) — 1 token per timestep.
        The depth transformer handles the rest at generation time.
        """
        device = text_tokens.device
        B = text_tokens.shape[0]
        parts = [text_tokens]

        if images is not None and self.image_tokenizer is not None:
            img_start = torch.full((B, 1), SPECIAL_TOKENS["image_start"], device=device)
            img_end = torch.full((B, 1), SPECIAL_TOKENS["image_end"], device=device)
            img_tokens = self.encode_image(images)
            parts.extend([img_start, img_tokens, img_end])

        if mel is not None and self.audio_tokenizer is not None:
            aud_start = torch.full((B, 1), SPECIAL_TOKENS["audio_start"], device=device)
            aud_end = torch.full((B, 1), SPECIAL_TOKENS["audio_end"], device=device)
            semantic_tokens = self.encode_audio_semantic(mel)
            parts.extend([aud_start, semantic_tokens, aud_end])

        return torch.cat(parts, dim=1)

    def build_duplex_input(
        self,
        text_tokens: torch.Tensor,
        user_mel: torch.Tensor,
        agent_mel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build full-duplex interleaved sequence (PersonaPlex-style).

        Returns:
            input_ids: (B, seq) interleaved user/agent semantic tokens
            agent_positions: (B, num_frames) positions of agent tokens in sequence
        """
        device = text_tokens.device
        B = text_tokens.shape[0]

        user_semantic = self.encode_audio_semantic(user_mel)  # (B, T)
        T = user_semantic.shape[1]

        parts = [text_tokens]
        parts.append(torch.full((B, 1), SPECIAL_TOKENS["audio_start"], device=device))

        agent_positions = []
        if agent_mel is not None:
            agent_semantic = self.encode_audio_semantic(agent_mel)
            T = min(T, agent_semantic.shape[1])
        else:
            agent_semantic = None

        # Interleave: user_t0, agent_t0, user_t1, agent_t1, ...
        current_pos = text_tokens.shape[1] + 1  # after text + audio_start
        for t in range(T):
            # User marker + semantic token
            parts.append(torch.full((B, 1), SPECIAL_TOKENS["user_audio"], device=device))
            parts.append(user_semantic[:, t:t+1])
            current_pos += 2

            # Agent marker + semantic token
            parts.append(torch.full((B, 1), SPECIAL_TOKENS["agent_audio"], device=device))
            if agent_semantic is not None:
                parts.append(agent_semantic[:, t:t+1])
            else:
                # Placeholder for generation
                parts.append(torch.full((B, 1), SPECIAL_TOKENS["agent_audio"], device=device))
            agent_positions.append(current_pos + 1)  # position of agent token
            current_pos += 2

        parts.append(torch.full((B, 1), SPECIAL_TOKENS["audio_end"], device=device))

        input_ids = torch.cat(parts, dim=1)
        agent_pos_tensor = torch.tensor(agent_positions, device=device).unsqueeze(0).expand(B, -1)

        return input_ids, agent_pos_tensor

    # ── Generation ──

    @torch.no_grad()
    def generate_image_tokens(
        self,
        prompt_ids: torch.Tensor,
        temperature: float = 0.9,
        top_k: int = 100,
    ) -> torch.Tensor:
        """Generate image VQ tokens autoregressively."""
        device = prompt_ids.device
        B = prompt_ids.shape[0]
        num_tokens = self.config.image_num_tokens

        gen_tok = torch.full((B, 1), SPECIAL_TOKENS["generate_image"], device=device)
        img_start = torch.full((B, 1), SPECIAL_TOKENS["image_start"], device=device)
        ids = torch.cat([prompt_ids, gen_tok, img_start], dim=1)

        img_offset = self.config.image_token_offset
        img_end = img_offset + self.config.image_codebook_size

        generated = []
        for _ in range(num_tokens):
            output = self.text_model.forward(ids)
            logits = output["logits"][:, -1, :]

            mask = torch.full_like(logits, float("-inf"))
            mask[:, img_offset:img_end] = 0
            logits = logits + mask

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token)
            ids = torch.cat([ids, next_token], dim=1)

        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate_audio_tokens(
        self,
        prompt_ids: torch.Tensor,
        max_audio_frames: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> list[torch.Tensor]:
        """Generate full audio: semantic tokens from temporal, acoustic from depth.

        Returns list of 8 tensors (B, T), one per codebook — ready for audio decoder.
        """
        device = prompt_ids.device
        B = prompt_ids.shape[0]

        gen_tok = torch.full((B, 1), SPECIAL_TOKENS["generate_audio"], device=device)
        aud_start = torch.full((B, 1), SPECIAL_TOKENS["audio_start"], device=device)
        ids = torch.cat([prompt_ids, gen_tok, aud_start], dim=1)

        sem_offset = self.config.audio_semantic_offset
        sem_end = sem_offset + self.config.audio_codebook_size

        all_semantic = []
        all_acoustic = [[] for _ in range(self.config.audio_num_codebooks - 1)]

        for _ in range(max_audio_frames):
            # Temporal: predict semantic token
            hidden_states = self.text_model.model(ids)
            logits = F.linear(hidden_states[:, -1:], self.text_model.get_lm_head_weight())
            logits = logits[:, 0, :]

            # Allow semantic tokens + audio_end
            mask = torch.full_like(logits, float("-inf"))
            mask[:, sem_offset:sem_end] = 0
            mask[:, SPECIAL_TOKENS["audio_end"]] = 0
            logits = logits + mask

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            if (next_token == SPECIAL_TOKENS["audio_end"]).any():
                break

            # Store semantic token (local codebook index)
            all_semantic.append(next_token[:, 0] - sem_offset)
            ids = torch.cat([ids, next_token], dim=1)

            # Depth: predict acoustic codebooks 1-7 from temporal hidden state
            frame_hidden = hidden_states[:, -1]  # (B, hidden)
            depth_result = self.depth_transformer(frame_hidden)
            for i, logits_i in enumerate(depth_result["logits"]):
                acoustic_token = logits_i.argmax(dim=-1)  # (B,)
                all_acoustic[i].append(acoustic_token)

        if not all_semantic:
            return [torch.zeros(B, 0, dtype=torch.long, device=device) for _ in range(self.config.audio_num_codebooks)]

        # Assemble all codebooks
        result = [torch.stack(all_semantic, dim=1)]  # codebook 0
        for acoustic_list in all_acoustic:
            result.append(torch.stack(acoustic_list, dim=1))
        return result

    @torch.no_grad()
    def generate_speech(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor | None:
        """Text -> speech mel spectrogram."""
        if self.audio_tokenizer is None:
            return None
        codebook_tokens = self.generate_audio_tokens(prompt_ids, **kwargs)
        if codebook_tokens[0].shape[1] == 0:
            return None
        return self.decode_audio(codebook_tokens)

    @torch.no_grad()
    def generate_image(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor | None:
        """Text -> image pixels."""
        if self.image_tokenizer is None:
            return None
        image_tokens = self.generate_image_tokens(prompt_ids, **kwargs)
        return self.decode_image(image_tokens)

    # ── Parameter counting ──

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        breakdown = {
            "temporal_transformer": sum(p.numel() for p in self.text_model.parameters()),
            "depth_transformer": sum(p.numel() for p in self.depth_transformer.parameters()),
        }
        if self.image_tokenizer:
            breakdown["image_tokenizer"] = sum(p.numel() for p in self.image_tokenizer.parameters())
        if self.audio_tokenizer:
            breakdown["audio_tokenizer"] = sum(p.numel() for p in self.audio_tokenizer.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
            "trainable_millions": trainable / 1e6,
            "breakdown": {k: v / 1e6 for k, v in breakdown.items()},
        }

    # ── Save / Load ──

    def save_pretrained(self, path: str):
        import json
        from pathlib import Path as P
        from safetensors.torch import save_file

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.text_model.save_pretrained(str(save_dir / "text"))
        save_file(self.depth_transformer.state_dict(), str(save_dir / "depth_transformer.safetensors"))

        if self.image_tokenizer:
            self.image_tokenizer.save_pretrained(str(save_dir / "image_tokenizer"))
        if self.audio_tokenizer:
            self.audio_tokenizer.save_pretrained(str(save_dir / "audio_tokenizer"))

        (save_dir / "multimodal_config.json").write_text(json.dumps({
            **self.config.to_dict(),
            "has_image_tokenizer": self.image_tokenizer is not None,
            "has_audio_tokenizer": self.audio_tokenizer is not None,
        }, indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TerraMultimodal":
        import json
        from pathlib import Path as P
        from safetensors.torch import load_file

        save_dir = P(path)
        mm_data = json.loads((save_dir / "multimodal_config.json").read_text())
        config = UnifiedMultimodalConfig.from_dict(mm_data)

        text_model = TerraForCausalLM.from_pretrained(str(save_dir / "text"), device=device)

        image_tokenizer = None
        if mm_data.get("has_image_tokenizer") and (save_dir / "image_tokenizer").exists():
            image_tokenizer = ImageTokenizer.from_pretrained(str(save_dir / "image_tokenizer"), device=device)

        audio_tokenizer = None
        if mm_data.get("has_audio_tokenizer") and (save_dir / "audio_tokenizer").exists():
            audio_tokenizer = AudioTokenizer.from_pretrained(str(save_dir / "audio_tokenizer"), device=device)

        model = cls(text_model, config, image_tokenizer, audio_tokenizer)

        depth_path = save_dir / "depth_transformer.safetensors"
        if depth_path.exists():
            depth_state = load_file(str(depth_path), device=device)
            model.depth_transformer.load_state_dict(depth_state)

        return model
