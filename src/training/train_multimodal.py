"""Training loops for all multimodal components.

Each component can be trained independently and in parallel with text pre-training.
Training order doesn't matter — they all project into the same embedding space.

Components:
1. Vision encoder — contrastive learning on image-caption pairs (SigLIP-style)
2. Image generator — VAE pre-training then diffusion training on image-caption pairs
3. Audio encoder (STT) — CTC loss on audio-transcript pairs
4. Speech decoder (TTS) — codec reconstruction loss on speech audio
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Vision Encoder Training (contrastive)
# ---------------------------------------------------------------------------

def train_vision_encoder(
    data_dir: str = "data/vision",
    output_dir: str = "models/checkpoints/vision",
    preset: str = "vision_tiny",
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_steps: int = 5000,
    warmup_steps: int = 200,
    eval_steps: int = 0,
    save_steps: int = 0,
    lm_hidden_size: int = 896,
) -> dict:
    """Train vision encoder with contrastive loss (SigLIP-style).

    The encoder learns to match images with their captions in embedding space.
    """
    from src.data.multimodal_downloader import VisionDataset
    from src.training.vision_encoder import TerraVisionEncoder, VisionConfig

    device = get_device()
    print(f"[vision] Device: {device}")

    # Config
    config_fn = getattr(VisionConfig, preset, None)
    if config_fn is None:
        raise ValueError(f"Unknown preset: {preset}")
    config = config_fn(projection_size=lm_hidden_size)

    # Model
    model = TerraVisionEncoder(config).to(device)
    params = model.count_parameters()
    print(f"[vision] Model: {preset} ({params['total_millions']:.1f}M params)")

    # Text projection for contrastive learning
    text_proj = nn.Linear(lm_hidden_size, lm_hidden_size).to(device)

    # Data
    dataset = VisionDataset(data_dir)
    print(f"[vision] Dataset: {len(dataset)} samples")

    val_size = max(1, int(len(dataset) * 0.05))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Auto steps
    steps_per_epoch = max(1, len(train_loader))
    if max_steps <= 0:
        max_steps = steps_per_epoch * 10
    if eval_steps <= 0:
        eval_steps = steps_per_epoch
    if save_steps <= 0:
        save_steps = eval_steps

    # We need a small text encoder to get caption embeddings for contrastive loss.
    # Use Terra's own embedding layer for this.
    from src.training.model import TerraConfig, TerraForCausalLM
    text_config = TerraConfig.terra_150m()
    text_embed = nn.Embedding(text_config.vocab_size, lm_hidden_size).to(device)

    # Optimizer
    all_params = list(model.parameters()) + list(text_proj.parameters()) + list(text_embed.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=0.01)

    # Temperature for contrastive loss
    log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    optimizer.add_param_group({"params": [log_temp]})

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[vision] Training: {max_steps} steps, eval every {eval_steps}")
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    while step < max_steps:
        model.train()
        for batch in train_loader:
            if step >= max_steps:
                break

            pixel_values = batch["pixel_values"].to(device)
            caption_ids = batch["caption_ids"].to(device)

            # Image embeddings: mean pool over patches
            image_embeds = model(pixel_values).mean(dim=1)  # (B, hidden)
            image_embeds = F.normalize(image_embeds, dim=-1)

            # Text embeddings: mean pool over tokens
            text_embeds = text_embed(caption_ids).mean(dim=1)  # (B, hidden)
            text_embeds = text_proj(text_embeds)
            text_embeds = F.normalize(text_embeds, dim=-1)

            # Contrastive loss (SigLIP-style symmetric)
            temp = log_temp.exp().clamp(max=100)
            logits = image_embeds @ text_embeds.T * temp
            labels = torch.arange(logits.shape[0], device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{max_steps} | loss {loss.item():.4f} | temp {temp.item():.2f}")

            if step % eval_steps == 0:
                val_loss = _eval_vision(model, text_embed, text_proj, log_temp, val_loader, device)
                print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(str(out / "best"))
                    print(f"  New best model saved")

    elapsed = time.time() - start_time
    model.save_pretrained(str(out / "final"))
    result = {
        "total_steps": step,
        "best_val_loss": best_val_loss,
        "training_time_seconds": elapsed,
        "parameter_count": params["total"],
    }
    print(f"[vision] Training complete: {result}")
    return result


def _eval_vision(model, text_embed, text_proj, log_temp, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            caption_ids = batch["caption_ids"].to(device)

            image_embeds = F.normalize(model(pixel_values).mean(dim=1), dim=-1)
            text_embeds = F.normalize(text_proj(text_embed(caption_ids).mean(dim=1)), dim=-1)

            temp = log_temp.exp().clamp(max=100)
            logits = image_embeds @ text_embeds.T * temp
            labels = torch.arange(logits.shape[0], device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Image Generator Training (VAE + diffusion)
# ---------------------------------------------------------------------------

def train_image_generator(
    data_dir: str = "data/image_gen",
    output_dir: str = "models/checkpoints/image_gen",
    preset: str = "gen_tiny",
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 10000,
    vae_pretrain_steps: int = 2000,
    warmup_steps: int = 200,
    lm_hidden_size: int = 896,
) -> dict:
    """Train image generator: first VAE, then diffusion U-Net."""
    from src.data.multimodal_downloader import VisionDataset
    from src.training.image_generator import ImageGenConfig, TerraImageGenerator

    device = get_device()
    print(f"[image_gen] Device: {device}")

    config_fn = getattr(ImageGenConfig, preset, None)
    if config_fn is None:
        raise ValueError(f"Unknown preset: {preset}")
    config = config_fn(context_dim=lm_hidden_size)

    model = TerraImageGenerator(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[image_gen] Model: {preset} ({total_params / 1e6:.1f}M params)")

    dataset = VisionDataset(data_dir)
    print(f"[image_gen] Dataset: {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Text embedding for conditioning
    text_embed = nn.Embedding(32000, lm_hidden_size).to(device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Phase 1: Pre-train VAE
    print(f"\n[image_gen] Phase 1: Pre-training VAE ({vae_pretrain_steps} steps)...")
    vae_optimizer = torch.optim.AdamW(model.vae.parameters(), lr=learning_rate)
    step = 0
    while step < vae_pretrain_steps:
        for batch in loader:
            if step >= vae_pretrain_steps:
                break
            images = batch["pixel_values"].to(device)
            # Resize to generator's expected size
            if images.shape[-1] != config.image_size:
                images = F.interpolate(images, size=config.image_size, mode="bilinear", align_corners=False)

            result = model.train_vae_step(images)
            vae_optimizer.zero_grad()
            result["loss"].backward()
            vae_optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"  [VAE] step {step}/{vae_pretrain_steps} | loss {result['loss'].item():.4f}")

    # Phase 2: Train diffusion U-Net
    print(f"\n[image_gen] Phase 2: Training diffusion ({max_steps - vae_pretrain_steps} steps)...")
    # Freeze VAE, train U-Net
    for p in model.vae.parameters():
        p.requires_grad = False
    unet_params = list(model.unet.parameters()) + list(text_embed.parameters())
    unet_optimizer = torch.optim.AdamW(unet_params, lr=learning_rate)

    diff_step = 0
    diff_max = max_steps - vae_pretrain_steps
    while diff_step < diff_max:
        for batch in loader:
            if diff_step >= diff_max:
                break
            images = batch["pixel_values"].to(device)
            caption_ids = batch["caption_ids"].to(device)

            if images.shape[-1] != config.image_size:
                images = F.interpolate(images, size=config.image_size, mode="bilinear", align_corners=False)

            # Get text conditioning
            context = text_embed(caption_ids)  # (B, seq, hidden)

            result = model.training_step(images, context)
            unet_optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(unet_params, 1.0)
            unet_optimizer.step()

            diff_step += 1
            if diff_step % 10 == 0:
                print(f"  [Diffusion] step {diff_step}/{diff_max} | loss {result['loss'].item():.4f}")

    # Save
    torch.save(model.state_dict(), str(out / "image_generator.pt"))
    with open(out / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    total_steps = vae_pretrain_steps + diff_step
    result = {"total_steps": total_steps, "parameter_count": total_params}
    print(f"[image_gen] Training complete: {result}")
    return result


# ---------------------------------------------------------------------------
# Audio Encoder Training (CTC for STT)
# ---------------------------------------------------------------------------

def train_audio_encoder(
    data_dir: str = "data/audio",
    output_dir: str = "models/checkpoints/audio",
    preset: str = "audio_tiny",
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_steps: int = 5000,
    warmup_steps: int = 200,
    eval_steps: int = 0,
    lm_hidden_size: int = 896,
) -> dict:
    """Train audio encoder with CTC loss for speech-to-text."""
    from src.data.multimodal_downloader import AudioDataset
    from src.training.audio_encoder import AudioConfig, TerraAudioEncoder

    device = get_device()
    print(f"[audio] Device: {device}")

    config_fn = getattr(AudioConfig, preset, None)
    if config_fn is None:
        raise ValueError(f"Unknown preset: {preset}")
    config = config_fn(projection_size=lm_hidden_size)

    model = TerraAudioEncoder(config).to(device)
    params = model.count_parameters()
    print(f"[audio] Model: {preset} ({params['total_millions']:.1f}M params)")

    dataset = AudioDataset(data_dir)
    print(f"[audio] Dataset: {len(dataset)} samples")

    val_size = max(1, int(len(dataset) * 0.05))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    steps_per_epoch = max(1, len(train_loader))
    if max_steps <= 0:
        max_steps = steps_per_epoch * 10
    if eval_steps <= 0:
        eval_steps = steps_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[audio] Training: {max_steps} steps, eval every {eval_steps}")
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    while step < max_steps:
        model.train()
        for batch in train_loader:
            if step >= max_steps:
                break

            mel = batch["mel"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_len = batch["text_len"]

            # CTC forward
            log_probs = model.forward_ctc(mel)  # (B, T, vocab)
            log_probs = log_probs.permute(1, 0, 2)  # (T, B, vocab) for CTC

            input_lengths = torch.full((mel.shape[0],), log_probs.shape[0], dtype=torch.long)
            target_lengths = text_len.long()

            loss = ctc_loss_fn(log_probs, text_ids, input_lengths, target_lengths)

            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{max_steps} | loss {loss.item():.4f}")

            if step % eval_steps == 0:
                val_loss = _eval_audio(model, ctc_loss_fn, val_loader, device)
                print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(str(out / "best"))
                    print(f"  New best model saved")

    elapsed = time.time() - start_time
    model.save_pretrained(str(out / "final"))
    result = {
        "total_steps": step,
        "best_val_loss": best_val_loss,
        "training_time_seconds": elapsed,
        "parameter_count": params["total"],
    }
    print(f"[audio] Training complete: {result}")
    return result


def _eval_audio(model, ctc_loss_fn, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_len = batch["text_len"]

            log_probs = model.forward_ctc(mel).permute(1, 0, 2)
            input_lengths = torch.full((mel.shape[0],), log_probs.shape[0], dtype=torch.long)
            target_lengths = text_len.long()

            loss = ctc_loss_fn(log_probs, text_ids, input_lengths, target_lengths)
            if torch.isfinite(loss):
                total_loss += loss.item()
                count += 1
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Speech Decoder Training (codec reconstruction)
# ---------------------------------------------------------------------------

def train_speech_decoder(
    data_dir: str = "data/tts",
    output_dir: str = "models/checkpoints/speech",
    preset: str = "speech_tiny",
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_steps: int = 5000,
    warmup_steps: int = 200,
    eval_steps: int = 0,
    lm_hidden_size: int = 896,
) -> dict:
    """Train speech decoder codec on mel reconstruction."""
    from src.data.multimodal_downloader import AudioDataset
    from src.training.speech_decoder import SpeechDecoderConfig, TerraSpeechDecoder

    device = get_device()
    print(f"[speech] Device: {device}")

    config_fn = getattr(SpeechDecoderConfig, preset, None)
    if config_fn is None:
        raise ValueError(f"Unknown preset: {preset}")
    config = config_fn(lm_hidden_size=lm_hidden_size)

    model = TerraSpeechDecoder(config).to(device)
    params = model.count_parameters()
    print(f"[speech] Model: {preset} ({params['total_millions']:.1f}M params)")

    dataset = AudioDataset(data_dir)
    print(f"[speech] Dataset: {len(dataset)} samples")

    val_size = max(1, int(len(dataset) * 0.05))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    steps_per_epoch = max(1, len(train_loader))
    if max_steps <= 0:
        max_steps = steps_per_epoch * 10
    if eval_steps <= 0:
        eval_steps = steps_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[speech] Training: {max_steps} steps, eval every {eval_steps}")
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    while step < max_steps:
        model.train()
        for batch in train_loader:
            if step >= max_steps:
                break

            mel = batch["mel"].to(device)

            # Codec forward: encode -> VQ -> decode -> reconstruct
            mel_hat, token_ids, commit_loss = model.forward_codec(mel)

            # Reconstruction loss
            recon_loss = F.l1_loss(mel_hat, mel)
            loss = recon_loss + 0.25 * commit_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{max_steps} | loss {loss.item():.4f} | recon {recon_loss.item():.4f} | commit {commit_loss.item():.4f}")

            if step % eval_steps == 0:
                val_loss = _eval_speech(model, val_loader, device)
                print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(str(out / "best"))
                    print(f"  New best model saved")

    elapsed = time.time() - start_time
    model.save_pretrained(str(out / "final"))
    result = {
        "total_steps": step,
        "best_val_loss": best_val_loss,
        "training_time_seconds": elapsed,
        "parameter_count": params["total"],
    }
    print(f"[speech] Training complete: {result}")
    return result


def _eval_speech(model, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            mel_hat, _, commit_loss = model.forward_codec(mel)
            loss = F.l1_loss(mel_hat, mel) + 0.25 * commit_loss
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)
