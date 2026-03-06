"""Terra: Custom transformer architecture for from-scratch pre-training.

Starts small (~150M params), scales to 3B. Designed for consumer hardware.
Key features:
- Grouped Query Attention (GQA) for memory efficiency
- RoPE positional encoding for length generalization
- SwiGLU activation (better than GELU for small models)
- RMSNorm (faster than LayerNorm)
- Sliding window attention option for long contexts
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TerraConfig:
    """Model configuration. Presets for different scales."""

    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4  # GQA: fewer KV heads saves memory
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    @classmethod
    def terra_150m(cls) -> "TerraConfig":
        """~150M params - trainable on MacBook Air M4 24GB."""
        return cls(
            hidden_size=896,
            intermediate_size=2432,
            num_hidden_layers=12,
            num_attention_heads=14,
            num_key_value_heads=2,
        )

    @classmethod
    def terra_400m(cls) -> "TerraConfig":
        """~400M params - cloud GPU burst."""
        return cls(
            hidden_size=1280,
            intermediate_size=3456,
            num_hidden_layers=20,
            num_attention_heads=20,
            num_key_value_heads=4,
        )

    @classmethod
    def terra_1b(cls) -> "TerraConfig":
        """~1B params - cloud GPU."""
        return cls(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
        )

    @classmethod
    def terra_3b(cls) -> "TerraConfig":
        """~3B params - final target."""
        return cls(
            hidden_size=2560,
            intermediate_size=6912,
            num_hidden_layers=36,
            num_attention_heads=20,
            num_key_value_heads=4,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "TerraConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def param_count_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.hidden_size
        attn_per_layer = (
            self.hidden_size * self.hidden_size  # Q
            + self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads)  # K
            + self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads)  # V
            + self.hidden_size * self.hidden_size  # O
        )
        ffn_per_layer = 3 * self.hidden_size * self.intermediate_size  # gate + up + down
        norm_per_layer = 2 * self.hidden_size
        total = embed + self.num_hidden_layers * (attn_per_layer + ffn_per_layer + norm_per_layer) + self.hidden_size
        if not self.tie_word_embeddings:
            total += embed
        return total


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


def _compute_rope_freqs(dim: int, max_pos: int, theta: float = 500000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.to(x.dtype)


class TerraAttention(nn.Module):
    def __init__(self, config: TerraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_freqs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = _apply_rope(q, rope_freqs)
        k = _apply_rope(k, rope_freqs)

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class TerraMLP(nn.Module):
    """SwiGLU FFN - better than standard GELU for small models."""

    def __init__(self, config: TerraConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TerraBlock(nn.Module):
    def __init__(self, config: TerraConfig):
        super().__init__()
        self.attention = TerraAttention(config)
        self.mlp = TerraMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_freqs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm residual connections
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, rope_freqs, attention_mask)
        hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TerraModel(nn.Module):
    """Terra transformer: custom architecture for from-scratch pre-training."""

    def __init__(self, config: TerraConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TerraBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            _compute_rope_freqs(
                config.hidden_size // config.num_attention_heads,
                config.max_position_embeddings,
                config.rope_theta,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, self.rope_freqs, attention_mask)

        return self.norm(hidden_states)


class TerraForCausalLM(nn.Module):
    """Terra model with language modeling head."""

    def __init__(self, config: TerraConfig):
        super().__init__()
        self.config = config
        self.model = TerraModel(config)

        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 / Llama conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_lm_head_weight(self) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head.weight
        return self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        hidden_states = self.model(input_ids, attention_mask)
        logits = F.linear(hidden_states, self.get_lm_head_weight())

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Simple autoregressive generation with repetition penalty."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Only use last max_position_embeddings tokens
                idx_cond = input_ids[:, -self.config.max_position_embeddings:]
                output = self.forward(idx_cond)
                logits = output["logits"][:, -1, :]

                # Repetition penalty: reduce score of tokens already generated
                if repetition_penalty != 1.0:
                    for token_id in set(input_ids[0].tolist()):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

                if temperature > 0:
                    logits = logits / temperature
                    # Top-k filtering
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    # Top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Stop on EOS (token 2 by convention)
                if next_token.item() == 2:
                    break

        return input_ids

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
        }

    def save_pretrained(self, path: str):
        """Save model weights and config."""
        import json
        from pathlib import Path as P
        from safetensors.torch import save_file

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(save_dir / "model.safetensors"))
        (save_dir / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TerraForCausalLM":
        """Load model from saved weights."""
        import json
        from pathlib import Path as P
        from safetensors.torch import load_file

        config_data = json.loads((P(path) / "config.json").read_text())
        config = TerraConfig.from_dict(config_data)
        model = cls(config)
        state_dict = load_file(str(P(path) / "model.safetensors"), device=device)
        model.load_state_dict(state_dict)
        return model
