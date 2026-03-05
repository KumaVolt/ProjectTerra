"""Terra Multimodal Model — unifies text, vision, audio in a single model.

This is the top-level model that combines:
- TerraForCausalLM (text backbone)
- TerraVisionEncoder (image understanding)
- TerraAudioEncoder (speech-to-text)
- TerraSpeechDecoder (text-to-speech)
- TerraImageGenerator (text-to-image via latent diffusion)

All modalities project into the same embedding space (Terra's hidden_size).
The LLM processes a unified sequence of [text tokens | image tokens | audio tokens].

Special tokens mark modality boundaries:
- <|image_start|> ... <|image_end|> for vision
- <|audio_start|> ... <|audio_end|> for audio

Each encoder can be trained independently, then plugged in.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.model import TerraConfig, TerraForCausalLM


@dataclass
class MultimodalConfig:
    """Configuration for which modalities are active."""

    text: bool = True
    vision: bool = False
    image_generation: bool = False  # Text-to-image
    audio_input: bool = False  # STT
    audio_output: bool = False  # TTS
    full_duplex: bool = False

    # Freeze the text backbone when training modality adapters
    freeze_text_backbone: bool = True
    # Freeze vision/audio encoders (only train projections)
    freeze_encoders: bool = False

    def active_modalities(self) -> list[str]:
        mods = ["text"]
        if self.vision:
            mods.append("vision")
        if self.image_generation:
            mods.append("image_generation")
        if self.audio_input:
            mods.append("audio_input")
        if self.audio_output:
            mods.append("audio_output")
        if self.full_duplex:
            mods.append("full_duplex")
        return mods


# Special token IDs (must match tokenizer)
IMAGE_START_TOKEN = 8   # <|image_start|>
IMAGE_END_TOKEN = 9     # <|image_end|>
AUDIO_START_TOKEN = 10  # <|audio_start|>
AUDIO_END_TOKEN = 11    # <|audio_end|>


class TerraMultimodal(nn.Module):
    """Unified multimodal model.

    Processes interleaved sequences like:
        [text] [<|image_start|>] [image_tokens...] [<|image_end|>] [text] [<|audio_start|>] [audio_tokens...] [<|audio_end|>] [text]

    Each encoder independently converts its modality into embeddings
    that live in the same vector space as text token embeddings.
    """

    def __init__(
        self,
        text_model: TerraForCausalLM,
        vision_encoder: nn.Module | None = None,
        audio_encoder: nn.Module | None = None,
        speech_decoder: nn.Module | None = None,
        image_generator: nn.Module | None = None,
        config: MultimodalConfig | None = None,
    ):
        super().__init__()
        self.config = config or MultimodalConfig()
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.speech_decoder = speech_decoder
        self.image_generator = image_generator

        if self.config.freeze_text_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False

        if self.config.freeze_encoders:
            for encoder in [self.vision_encoder, self.audio_encoder]:
                if encoder is not None:
                    for param in encoder.parameters():
                        param.requires_grad = False

    def _build_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        mel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Replace special token embeddings with encoder outputs.

        The LLM's embedding layer creates initial embeddings for all tokens.
        We then replace the placeholder tokens between modality markers
        with actual encoder outputs.
        """
        # Start with text embeddings for everything
        embeddings = self.text_model.model.embed_tokens(input_ids)

        if pixel_values is not None and self.vision_encoder is not None:
            # Get vision embeddings
            vision_embeds = self.vision_encoder(pixel_values)  # (B, num_patches, hidden)

            # Find image token positions and replace
            for b in range(input_ids.shape[0]):
                img_starts = (input_ids[b] == IMAGE_START_TOKEN).nonzero(as_tuple=True)[0]
                img_ends = (input_ids[b] == IMAGE_END_TOKEN).nonzero(as_tuple=True)[0]

                for start, end in zip(img_starts, img_ends):
                    # Replace tokens between start+1 and end with vision embeddings
                    num_slots = end - start - 1
                    vis = vision_embeds[b]
                    if vis.shape[0] > num_slots:
                        vis = vis[:num_slots]  # Truncate if needed
                    elif vis.shape[0] < num_slots:
                        # Pad with last embedding
                        pad = vis[-1:].expand(num_slots - vis.shape[0], -1)
                        vis = torch.cat([vis, pad], dim=0)
                    embeddings[b, start + 1:end] = vis

        if mel_values is not None and self.audio_encoder is not None:
            # Get audio embeddings
            audio_embeds = self.audio_encoder(mel_values)  # (B, num_frames, hidden)

            for b in range(input_ids.shape[0]):
                aud_starts = (input_ids[b] == AUDIO_START_TOKEN).nonzero(as_tuple=True)[0]
                aud_ends = (input_ids[b] == AUDIO_END_TOKEN).nonzero(as_tuple=True)[0]

                for start, end in zip(aud_starts, aud_ends):
                    num_slots = end - start - 1
                    aud = audio_embeds[b]
                    if aud.shape[0] > num_slots:
                        aud = aud[:num_slots]
                    elif aud.shape[0] < num_slots:
                        pad = aud[-1:].expand(num_slots - aud.shape[0], -1)
                        aud = torch.cat([aud, pad], dim=0)
                    embeddings[b, start + 1:end] = aud

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        mel_values: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass with optional multimodal inputs.

        If no images/audio provided, this is equivalent to the text-only model.
        """
        has_multimodal = pixel_values is not None or mel_values is not None

        if has_multimodal:
            # Build mixed embeddings
            embeddings = self._build_multimodal_embeddings(input_ids, pixel_values, mel_values)
            # Forward through transformer layers (bypass embedding lookup)
            hidden_states = embeddings
            for layer in self.text_model.model.layers:
                hidden_states = layer(hidden_states, self.text_model.model.rope_freqs, attention_mask)
            hidden_states = self.text_model.model.norm(hidden_states)
            logits = F.linear(hidden_states, self.text_model.get_lm_head_weight())

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.text_model.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            return {"loss": loss, "logits": logits, "hidden_states": hidden_states}
        else:
            # Pure text — delegate to text model directly
            return self.text_model(input_ids, attention_mask, labels)

    def generate_image(
        self,
        input_ids: torch.Tensor,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor | None:
        """Generate an image from text input.

        Args:
            input_ids: (B, seq) text token IDs describing the image
            num_steps: denoising steps
            guidance_scale: how strongly to follow the text (higher = more faithful)
        Returns:
            (B, 3, H, W) generated images in [-1, 1], or None if no image generator
        """
        if self.image_generator is None:
            return None

        # Get text embeddings from LLM
        with torch.no_grad():
            hidden_states = self.text_model.model(input_ids)
            # Use all hidden states as conditioning context

        return self.image_generator.generate(hidden_states, num_steps, guidance_scale)

    def generate_speech(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Generate speech output from text (+ optional image) input.

        Returns mel spectrogram that can be converted to audio with a vocoder.
        """
        if self.speech_decoder is None:
            return None

        # Get LLM hidden states
        with torch.no_grad():
            result = self.forward(input_ids, pixel_values=pixel_values)
            hidden_states = result["hidden_states"]

        # Use speech decoder conditioned on LLM states
        # (In full pipeline, LLM would generate audio tokens; this is simplified)
        # For now, use the conditioner to generate mel from hidden states
        target_len = hidden_states.shape[1] * 4  # Rough upsampling
        conditioning = self.speech_decoder.lm_conditioner(hidden_states, target_len)
        mel = self.speech_decoder.codec_decoder(conditioning)
        return mel

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        breakdown = {"text": sum(p.numel() for p in self.text_model.parameters())}
        if self.vision_encoder:
            breakdown["vision"] = sum(p.numel() for p in self.vision_encoder.parameters())
        if self.audio_encoder:
            breakdown["audio_encoder"] = sum(p.numel() for p in self.audio_encoder.parameters())
        if self.speech_decoder:
            breakdown["speech_decoder"] = sum(p.numel() for p in self.speech_decoder.parameters())
        if self.image_generator:
            breakdown["image_generator"] = sum(p.numel() for p in self.image_generator.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
            "trainable_millions": trainable / 1e6,
            "breakdown": {k: v / 1e6 for k, v in breakdown.items()},
        }

    def save_pretrained(self, path: str):
        """Save all components."""
        import json
        from pathlib import Path as P

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save text model
        self.text_model.save_pretrained(str(save_dir / "text"))

        # Save encoders/decoders if present
        if self.vision_encoder:
            self.vision_encoder.save_pretrained(str(save_dir / "vision"))
        if self.audio_encoder:
            self.audio_encoder.save_pretrained(str(save_dir / "audio_encoder"))
        if self.speech_decoder:
            self.speech_decoder.save_pretrained(str(save_dir / "speech_decoder"))
        if self.image_generator:
            self.image_generator.save_pretrained(str(save_dir / "image_generator"))

        # Save multimodal config
        (save_dir / "multimodal_config.json").write_text(json.dumps({
            "modalities": self.config.active_modalities(),
            "has_vision": self.vision_encoder is not None,
            "has_audio_input": self.audio_encoder is not None,
            "has_audio_output": self.speech_decoder is not None,
            "has_image_generation": self.image_generator is not None,
        }, indent=2))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "TerraMultimodal":
        """Load multimodal model from saved components."""
        import json
        from pathlib import Path as P

        save_dir = P(path)
        mm_config = json.loads((save_dir / "multimodal_config.json").read_text())

        text_model = TerraForCausalLM.from_pretrained(str(save_dir / "text"), device=device)

        vision_encoder = None
        if mm_config.get("has_vision") and (save_dir / "vision").exists():
            from src.training.vision_encoder import TerraVisionEncoder
            vision_encoder = TerraVisionEncoder.from_pretrained(str(save_dir / "vision"), device=device)

        audio_encoder = None
        if mm_config.get("has_audio_input") and (save_dir / "audio_encoder").exists():
            from src.training.audio_encoder import TerraAudioEncoder
            audio_encoder = TerraAudioEncoder.from_pretrained(str(save_dir / "audio_encoder"), device=device)

        speech_decoder = None
        if mm_config.get("has_audio_output") and (save_dir / "speech_decoder").exists():
            from src.training.speech_decoder import TerraSpeechDecoder
            speech_decoder = TerraSpeechDecoder.from_pretrained(str(save_dir / "speech_decoder"), device=device)

        image_generator = None
        if mm_config.get("has_image_generation") and (save_dir / "image_generator").exists():
            from src.training.image_generator import TerraImageGenerator
            image_generator = TerraImageGenerator.from_pretrained(str(save_dir / "image_generator"), device=device)

        config = MultimodalConfig(
            vision=vision_encoder is not None,
            image_generation=image_generator is not None,
            audio_input=audio_encoder is not None,
            audio_output=speech_decoder is not None,
        )

        return cls(text_model, vision_encoder, audio_encoder, speech_decoder, image_generator, config)
