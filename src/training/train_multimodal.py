"""Training loops for all multimodal components (unified tokenizer architecture).

Components train independently, then plug into the unified transformer:
1. Image tokenizer (VQ-VAE) — image reconstruction loss
2. Audio tokenizer (codec) — mel reconstruction loss
3. Unified multimodal fine-tuning — next-token prediction on interleaved sequences

Legacy training loops for the old separate encoder architecture are kept
for backward compatibility (vision encoder contrastive, audio encoder CTC).
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
# Image Tokenizer Training (VQ-VAE reconstruction)
# ---------------------------------------------------------------------------

def train_image_tokenizer(
    data_dir: str = "data/vision",
    output_dir: str = "models/checkpoints/image_tokenizer",
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    max_steps: int = 10000,
    eval_steps: int = 0,
    image_size: int = 256,
    codebook_size: int = 8192,
    warmup_steps: int = 200,
) -> dict:
    """Train image VQ-VAE tokenizer on image reconstruction.

    The tokenizer learns to compress images into discrete tokens and
    reconstruct them. Once trained, it provides the image vocabulary
    for the unified transformer.
    """
    from src.data.multimodal_downloader import VisionDataset
    from src.training.multimodal import ImageTokenizer, UnifiedMultimodalConfig

    device = get_device()
    print(f"[image_tokenizer] Device: {device}")

    config = UnifiedMultimodalConfig(
        image_size=image_size,
        image_codebook_size=codebook_size,
        image_num_tokens=(image_size // 16) ** 2,
    )

    model = ImageTokenizer(config).to(device)
    params = model.count_parameters()
    print(f"[image_tokenizer] {params['total_millions']:.1f}M params, codebook={codebook_size}")

    dataset = VisionDataset(data_dir)
    print(f"[image_tokenizer] Dataset: {len(dataset)} samples")

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

    # Warmup + cosine decay scheduler to prevent loss explosion
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[image_tokenizer] Training: {max_steps} steps, lr={learning_rate}, warmup={warmup_steps}, eval every {eval_steps}")
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    while step < max_steps:
        model.train()
        for batch in train_loader:
            if step >= max_steps:
                break

            images = batch["pixel_values"].to(device)
            if images.shape[-1] != image_size:
                images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)

            reconstructed, token_ids, loss = model(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{max_steps} | loss {loss.item():.4f}")

            if step % eval_steps == 0:
                val_loss = _eval_image_tokenizer(model, val_loader, device, image_size)
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
    print(f"[image_tokenizer] Training complete: {result}")
    return result


def _eval_image_tokenizer(model, val_loader, device, image_size):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(device)
            if images.shape[-1] != image_size:
                images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
            _, _, loss = model(images)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Audio Tokenizer Training (codec reconstruction)
# ---------------------------------------------------------------------------

def train_audio_tokenizer(
    data_dir: str = "data/tts",
    output_dir: str = "models/checkpoints/audio_tokenizer",
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 5000,
    eval_steps: int = 0,
    warmup_steps: int = 100,
) -> dict:
    """Train audio codec tokenizer on mel reconstruction.

    The tokenizer learns to compress mel spectrograms into discrete tokens
    (via residual VQ) and reconstruct them. Once trained, it provides the
    audio vocabulary for the unified transformer.
    """
    from src.data.multimodal_downloader import AudioDataset
    from src.training.multimodal import AudioTokenizer, UnifiedMultimodalConfig

    device = get_device()
    print(f"[audio_tokenizer] Device: {device}")

    config = UnifiedMultimodalConfig()
    model = AudioTokenizer(config).to(device)
    params = model.count_parameters()
    print(f"[audio_tokenizer] {params['total_millions']:.1f}M params")

    dataset = AudioDataset(data_dir)
    print(f"[audio_tokenizer] Dataset: {len(dataset)} samples")

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

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[audio_tokenizer] Training: {max_steps} steps, lr={learning_rate}, warmup={warmup_steps}, eval every {eval_steps}")
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    while step < max_steps:
        model.train()
        for batch in train_loader:
            if step >= max_steps:
                break

            mel = batch["mel"].to(device)
            mel_hat, _token_ids, loss = model(mel)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{max_steps} | loss {loss.item():.4f}")

            if step % eval_steps == 0:
                val_loss = _eval_audio_tokenizer(model, val_loader, device)
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
    print(f"[audio_tokenizer] Training complete: {result}")
    return result


def _eval_audio_tokenizer(model, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            _, _, loss = model(mel)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Multimodal Fine-tuning (unified model: temporal + depth transformer)
# ---------------------------------------------------------------------------

class MultimodalDataset(torch.utils.data.Dataset):
    """Dataset that creates interleaved multimodal token sequences.

    Loads vision data (image + caption) and audio data (mel + transcript),
    then builds training sequences:
        - Image understanding: "Describe this image" <|img_start|> [tokens] <|img_end|> "caption"
        - Image generation:    "Draw: caption" <|gen_image|> <|img_start|> [tokens] <|img_end|>
        - Speech understanding: <|audio_start|> [semantic_tokens] <|audio_end|> "transcript"
        - Speech generation:   "Say: transcript" <|gen_audio|> <|audio_start|> [semantic_tokens] <|audio_end|>

    Each sample is a (input_ids, labels, audio_positions, acoustic_targets) tuple.
    """

    def __init__(
        self,
        vision_dir: str | None = "data/vision",
        audio_dir: str | None = "data/tts",
        image_tokenizer_path: str | None = None,
        audio_tokenizer_path: str | None = None,
        tokenizer_path: str = "models/tokenizer",
        max_seq_len: int = 512,
    ):
        from src.training.multimodal import (
            AudioTokenizer, ImageTokenizer, SPECIAL_TOKENS, UnifiedMultimodalConfig,
        )
        from src.training.tokenizer import load_tokenizer

        self.max_seq_len = max_seq_len
        self.text_tokenizer = load_tokenizer(tokenizer_path)
        self.special = SPECIAL_TOKENS

        # Load multimodal tokenizers if available
        self.image_tokenizer = None
        if image_tokenizer_path and Path(image_tokenizer_path).exists():
            self.image_tokenizer = ImageTokenizer.from_pretrained(image_tokenizer_path)
            self.image_tokenizer.eval()

        self.audio_tokenizer = None
        if audio_tokenizer_path and Path(audio_tokenizer_path).exists():
            self.audio_tokenizer = AudioTokenizer.from_pretrained(audio_tokenizer_path)
            self.audio_tokenizer.eval()

        # Load data manifests
        self.vision_entries = []
        if vision_dir and Path(vision_dir, "captions.jsonl").exists():
            with open(Path(vision_dir, "captions.jsonl")) as f:
                for line in f:
                    self.vision_entries.append(json.loads(line))

        self.audio_entries = []
        if audio_dir and Path(audio_dir, "manifest.jsonl").exists():
            with open(Path(audio_dir, "manifest.jsonl")) as f:
                for line in f:
                    self.audio_entries.append(json.loads(line))

        self.total = len(self.vision_entries) * 2 + len(self.audio_entries) * 2
        if self.total == 0:
            raise ValueError("No multimodal data found. Download with: terra download-multimodal")

        print(f"[mm_dataset] {len(self.vision_entries)} vision + {len(self.audio_entries)} audio entries = {self.total} training samples")

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> dict:
        n_vision = len(self.vision_entries) * 2
        if idx < n_vision:
            return self._build_vision_sample(idx)
        else:
            return self._build_audio_sample(idx - n_vision)

    def _tokenize_text(self, text: str, max_len: int = 64) -> list[int]:
        ids = self.text_tokenizer.encode(text).ids[:max_len]
        return ids

    @torch.no_grad()
    def _build_vision_sample(self, idx: int) -> dict:
        entry_idx = idx // 2
        is_generation = idx % 2 == 1  # alternate understanding / generation
        entry = self.vision_entries[entry_idx]

        caption = entry["caption"]
        caption_ids = self._tokenize_text(caption, max_len=64)

        # Get image tokens
        if self.image_tokenizer is not None:
            image = torch.load(entry["image"], weights_only=True)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] != self.image_tokenizer.config.image_size:
                image = F.interpolate(image, size=self.image_tokenizer.config.image_size,
                                      mode="bilinear", align_corners=False)
            img_tokens = self.image_tokenizer.encode(image)[0].tolist()
        else:
            # Dummy tokens if tokenizer not available (for testing pipeline)
            img_tokens = list(range(32000, 32000 + 16))

        S = self.special
        if is_generation:
            # "Draw: caption" <|gen_image|> <|img_start|> [image_tokens] <|img_end|>
            prefix = self._tokenize_text("Draw: " + caption, max_len=64)
            seq = prefix + [S["generate_image"], S["image_start"]] + img_tokens + [S["image_end"]]
            # Labels: predict image tokens (mask prefix)
            labels = [-100] * len(prefix) + [-100, -100] + img_tokens + [S["image_end"]]
        else:
            # <|img_start|> [image_tokens] <|img_end|> "caption"
            prompt = self._tokenize_text("Describe this image:", max_len=16)
            seq = prompt + [S["image_start"]] + img_tokens + [S["image_end"]] + caption_ids
            # Labels: predict caption (mask image input)
            n_prefix = len(prompt) + 1 + len(img_tokens) + 1
            labels = [-100] * n_prefix + caption_ids

        return self._pad_and_return(seq, labels)

    @torch.no_grad()
    def _build_audio_sample(self, idx: int) -> dict:
        entry_idx = idx // 2
        is_generation = idx % 2 == 1
        entry = self.audio_entries[entry_idx]

        text = entry["text"]
        text_ids = self._tokenize_text(text, max_len=64)

        # Get audio tokens
        semantic_tokens = []
        acoustic_tokens = None  # list of 7 lists for depth transformer
        if self.audio_tokenizer is not None:
            mel = torch.load(entry["mel"], weights_only=True)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            all_codebooks = self.audio_tokenizer.encode(mel)
            sem_offset = self.audio_tokenizer.config.audio_semantic_offset
            semantic_tokens = (all_codebooks[0][0] + sem_offset).tolist()
            # Limit length
            max_audio = (self.max_seq_len - len(text_ids) - 10) // 1
            semantic_tokens = semantic_tokens[:max_audio]
            # Acoustic targets for depth transformer (codebooks 1-7)
            acoustic_tokens = [cb[0][:max_audio].tolist() for cb in all_codebooks[1:]]
        else:
            sem_offset = 40192
            semantic_tokens = list(range(sem_offset, sem_offset + 10))
            acoustic_tokens = [[0] * 10 for _ in range(7)]

        S = self.special
        if is_generation:
            # "Say: text" <|gen_audio|> <|audio_start|> [semantic_tokens] <|audio_end|>
            prefix = self._tokenize_text("Say: " + text, max_len=64)
            seq = prefix + [S["generate_audio"], S["audio_start"]] + semantic_tokens + [S["audio_end"]]
            n_prefix = len(prefix) + 2  # gen_audio + audio_start
            labels = [-100] * n_prefix + semantic_tokens + [S["audio_end"]]
        else:
            # <|audio_start|> [semantic_tokens] <|audio_end|> "text"
            seq = [S["audio_start"]] + semantic_tokens + [S["audio_end"]] + text_ids
            n_prefix = 1 + len(semantic_tokens) + 1
            labels = [-100] * n_prefix + text_ids

        return self._pad_and_return(seq, labels, semantic_tokens, acoustic_tokens, is_generation)

    def _pad_and_return(
        self,
        seq: list[int],
        labels: list[int],
        semantic_tokens: list[int] | None = None,
        acoustic_tokens: list[list[int]] | None = None,
        has_audio_gen: bool = False,
    ) -> dict:
        # Truncate
        seq = seq[:self.max_seq_len]
        labels = labels[:self.max_seq_len]

        # Pad
        pad_len = self.max_seq_len - len(seq)
        seq = seq + [0] * pad_len
        labels = labels + [-100] * pad_len

        result = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Audio positions and acoustic targets for depth transformer
        if has_audio_gen and semantic_tokens and acoustic_tokens:
            # Find positions of semantic audio tokens in the sequence
            sem_offset = self.audio_tokenizer.config.audio_semantic_offset if self.audio_tokenizer else 40192
            sem_end = sem_offset + 1024
            positions = [i for i, tok in enumerate(seq) if sem_offset <= tok < sem_end]
            n_frames = len(positions)
            result["audio_positions"] = torch.tensor(positions, dtype=torch.long)
            result["acoustic_targets"] = torch.tensor(
                [cb[:n_frames] for cb in acoustic_tokens], dtype=torch.long
            )  # (7, n_frames)
        else:
            result["audio_positions"] = torch.zeros(0, dtype=torch.long)
            result["acoustic_targets"] = torch.zeros(7, 0, dtype=torch.long)

        return result


def _mm_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length audio positions."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    # Pad audio positions to max length in batch
    max_audio_len = max(b["audio_positions"].shape[0] for b in batch)
    audio_positions = torch.zeros(len(batch), max(max_audio_len, 1), dtype=torch.long)
    acoustic_targets = torch.zeros(len(batch), 7, max(max_audio_len, 1), dtype=torch.long)
    has_audio = torch.zeros(len(batch), dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["audio_positions"].shape[0]
        if n > 0:
            audio_positions[i, :n] = b["audio_positions"]
            acoustic_targets[i, :, :n] = b["acoustic_targets"]
            has_audio[i] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "audio_positions": audio_positions,
        "acoustic_targets": acoustic_targets,
        "has_audio": has_audio,
    }


def train_multimodal(
    text_model_path: str = "models/current",
    image_tokenizer_path: str = "models/checkpoints/image_tokenizer/best",
    audio_tokenizer_path: str = "models/checkpoints/audio_tokenizer/best",
    vision_dir: str = "data/vision",
    audio_dir: str = "data/tts",
    output_dir: str = "models/checkpoints/multimodal",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    max_steps: int = 10000,
    warmup_steps: int = 200,
    eval_steps: int = 0,
    save_steps: int = 0,
    max_seq_len: int = 512,
    depth_loss_weight: float = 1.0,
) -> dict:
    """Multimodal fine-tuning: train unified model on interleaved text+image+audio.

    This trains:
    1. The temporal transformer to handle text, image, and audio semantic tokens
    2. The depth transformer to predict acoustic codebooks 1-7

    Prerequisites:
    - Pre-trained text model (terra pretrain / terra sft)
    - Trained image tokenizer (terra train-image-tok)
    - Trained audio tokenizer (terra train-audio-tok)
    - Downloaded multimodal data (terra download-multimodal)
    """
    from src.training.model import TerraForCausalLM
    from src.training.multimodal import (
        AudioTokenizer, ImageTokenizer, TerraMultimodal, UnifiedMultimodalConfig,
    )

    device = get_device()
    print(f"[multimodal] Device: {device}")

    # Load text model
    print(f"[multimodal] Loading text model from {text_model_path}...")
    text_model = TerraForCausalLM.from_pretrained(text_model_path, device=str(device))

    # Load tokenizers (try best, fall back to final)
    def _resolve_tok_path(base: str) -> str | None:
        for suffix in ["", "/best", "/final"]:
            p = base.rstrip("/") + suffix if suffix else base
            if Path(p).exists() and any(Path(p).glob("*_config.json")):
                return p
        parent = str(Path(base).parent)
        for sub in ["best", "final"]:
            p = f"{parent}/{sub}"
            if Path(p).exists() and any(Path(p).glob("*_config.json")):
                return p
        return None

    img_tok = None
    img_path = _resolve_tok_path(image_tokenizer_path)
    if img_path:
        print(f"[multimodal] Loading image tokenizer from {img_path}")
        img_tok = ImageTokenizer.from_pretrained(img_path, device=str(device))
        img_tok.eval()
        for p in img_tok.parameters():
            p.requires_grad = False

    aud_tok = None
    aud_path = _resolve_tok_path(audio_tokenizer_path)
    if aud_path:
        print(f"[multimodal] Loading audio tokenizer from {aud_path}")
        aud_tok = AudioTokenizer.from_pretrained(aud_path, device=str(device))
        aud_tok.eval()
        for p in aud_tok.parameters():
            p.requires_grad = False

    if img_tok is None and aud_tok is None:
        raise RuntimeError("Need at least one trained tokenizer. Run: terra train-image-tok or terra train-audio-tok")

    # Build unified model
    config = UnifiedMultimodalConfig()
    config.freeze_text_backbone = False  # fine-tune everything
    model = TerraMultimodal(text_model, config, img_tok, aud_tok).to(device)

    params = model.count_parameters()
    print(f"[multimodal] Model: {params['total_millions']:.1f}M total, {params['trainable_millions']:.1f}M trainable")
    for k, v in params["breakdown"].items():
        print(f"  {k}: {v:.2f}M")

    # Dataset
    dataset = MultimodalDataset(
        vision_dir=vision_dir if img_tok else None,
        audio_dir=audio_dir if aud_tok else None,
        image_tokenizer_path=image_tokenizer_path if img_tok else None,
        audio_tokenizer_path=audio_tokenizer_path if aud_tok else None,
        max_seq_len=max_seq_len,
    )

    val_size = max(1, int(len(dataset) * 0.05))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=_mm_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=_mm_collate_fn,
    )

    steps_per_epoch = max(1, len(train_loader))
    if max_steps <= 0:
        max_steps = steps_per_epoch * 5
    if eval_steps <= 0:
        eval_steps = min(steps_per_epoch, 500)
    if save_steps <= 0:
        save_steps = eval_steps

    # Optimizer: different LR for temporal (lower) vs depth (higher)
    temporal_params = list(model.text_model.parameters())
    depth_params = list(model.depth_transformer.parameters())
    optimizer = torch.optim.AdamW([
        {"params": temporal_params, "lr": learning_rate},
        {"params": depth_params, "lr": learning_rate * 3},
    ], weight_decay=0.01)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[multimodal] Training: {max_steps} steps, eval every {eval_steps}, depth_weight={depth_loss_weight}")
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    accum_loss = 0.0

    while step < max_steps:
        model.train()
        # Keep tokenizers frozen
        if img_tok:
            img_tok.eval()
        if aud_tok:
            aud_tok.eval()

        for batch in train_loader:
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            audio_positions = batch["audio_positions"].to(device)
            acoustic_targets = batch["acoustic_targets"].to(device)
            has_audio = batch["has_audio"]

            # Split acoustic targets into list of 7 tensors
            target_acs = None
            aud_pos = None
            if has_audio.any():
                # Only pass audio data for batches that have it
                aud_pos = audio_positions
                if acoustic_targets.shape[2] > 0:
                    target_acs = [acoustic_targets[:, i] for i in range(7)]

            result = model.forward_with_depth(input_ids, aud_pos, target_acs, labels)

            loss = result["loss"] if result["loss"] is not None else torch.tensor(0.0, device=device)
            # Weight depth loss
            if result["depth_loss"] is not None and result["temporal_loss"] is not None:
                loss = result["temporal_loss"] + depth_loss_weight * result["depth_loss"]

            loss = loss / gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                avg_loss = accum_loss / 10 * gradient_accumulation_steps
                t_loss = result["temporal_loss"].item() if result["temporal_loss"] is not None else 0
                d_loss = result["depth_loss"].item() if result["depth_loss"] is not None else 0
                print(f"  step {step}/{max_steps} | loss {avg_loss:.4f} | temporal {t_loss:.4f} | depth {d_loss:.4f}")
                accum_loss = 0.0

            if step % eval_steps == 0:
                val_loss = _eval_multimodal(model, val_loader, device, depth_loss_weight)
                print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(str(out / "best"))
                    print(f"  New best model saved")

            if step % save_steps == 0 and step % eval_steps != 0:
                model.save_pretrained(str(out / f"step-{step}"))

    elapsed = time.time() - start_time
    model.save_pretrained(str(out / "final"))

    # Copy best to models/current for easy access
    best_dir = out / "best"
    if best_dir.exists():
        import shutil
        current_dir = Path("models/current_multimodal")
        if current_dir.exists():
            shutil.rmtree(current_dir)
        shutil.copytree(str(best_dir), str(current_dir))
        print(f"[multimodal] Best model copied to {current_dir}")

    result = {
        "total_steps": step,
        "best_val_loss": best_val_loss,
        "training_time_seconds": elapsed,
        "model_path": str(out / "best"),
    }
    print(f"[multimodal] Training complete: {result}")
    return result


def _eval_multimodal(model, val_loader, device, depth_loss_weight):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            audio_positions = batch["audio_positions"].to(device)
            acoustic_targets = batch["acoustic_targets"].to(device)
            has_audio = batch["has_audio"]

            target_acs = None
            aud_pos = None
            if has_audio.any():
                aud_pos = audio_positions
                if acoustic_targets.shape[2] > 0:
                    target_acs = [acoustic_targets[:, i] for i in range(7)]

            result = model.forward_with_depth(input_ids, aud_pos, target_acs, labels)
            loss = torch.tensor(0.0, device=device)
            if result["temporal_loss"] is not None:
                loss = result["temporal_loss"]
            if result["depth_loss"] is not None:
                loss = loss + depth_loss_weight * result["depth_loss"]

            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Legacy: Vision Encoder Training (contrastive) — kept for backward compat
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
    """Train vision encoder with contrastive loss (SigLIP-style)."""
    from src.data.multimodal_downloader import VisionDataset
    from src.training.vision_encoder import TerraVisionEncoder, VisionConfig

    device = get_device()
    print(f"[vision] Device: {device}")

    config_fn = getattr(VisionConfig, preset, None)
    if config_fn is None:
        raise ValueError(f"Unknown preset: {preset}")
    config = config_fn(projection_size=lm_hidden_size)

    model = TerraVisionEncoder(config).to(device)
    params = model.count_parameters()
    print(f"[vision] Model: {preset} ({params['total_millions']:.1f}M params)")

    text_proj = nn.Linear(lm_hidden_size, lm_hidden_size).to(device)

    dataset = VisionDataset(data_dir)
    print(f"[vision] Dataset: {len(dataset)} samples")

    val_size = max(1, int(len(dataset) * 0.05))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    steps_per_epoch = max(1, len(train_loader))
    if max_steps <= 0:
        max_steps = steps_per_epoch * 10
    if eval_steps <= 0:
        eval_steps = steps_per_epoch
    if save_steps <= 0:
        save_steps = eval_steps

    from src.training.model import TerraConfig
    text_config = TerraConfig.terra_150m()
    text_embed = nn.Embedding(text_config.vocab_size, lm_hidden_size).to(device)

    all_params = list(model.parameters()) + list(text_proj.parameters()) + list(text_embed.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=0.01)

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

            image_embeds = model(pixel_values).mean(dim=1)
            image_embeds = F.normalize(image_embeds, dim=-1)

            text_embeds = text_embed(caption_ids).mean(dim=1)
            text_embeds = text_proj(text_embeds)
            text_embeds = F.normalize(text_embeds, dim=-1)

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
# Legacy: Image Generator Training (diffusion) — kept for backward compat
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

    text_embed = nn.Embedding(32000, lm_hidden_size).to(device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Phase 1: VAE
    print(f"\n[image_gen] Phase 1: Pre-training VAE ({vae_pretrain_steps} steps)...")
    vae_optimizer = torch.optim.AdamW(model.vae.parameters(), lr=learning_rate)
    step = 0
    while step < vae_pretrain_steps:
        for batch in loader:
            if step >= vae_pretrain_steps:
                break
            images = batch["pixel_values"].to(device)
            if images.shape[-1] != config.image_size:
                images = F.interpolate(images, size=config.image_size, mode="bilinear", align_corners=False)

            result = model.train_vae_step(images)
            vae_optimizer.zero_grad()
            result["loss"].backward()
            vae_optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"  [VAE] step {step}/{vae_pretrain_steps} | loss {result['loss'].item():.4f}")

    # Phase 2: Diffusion
    print(f"\n[image_gen] Phase 2: Training diffusion ({max_steps - vae_pretrain_steps} steps)...")
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

            context = text_embed(caption_ids)
            result = model.training_step(images, context)
            unet_optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(unet_params, 1.0)
            unet_optimizer.step()

            diff_step += 1
            if diff_step % 10 == 0:
                print(f"  [Diffusion] step {diff_step}/{diff_max} | loss {result['loss'].item():.4f}")

    torch.save(model.state_dict(), str(out / "image_generator.pt"))
    with open(out / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    total_steps = vae_pretrain_steps + diff_step
    result = {"total_steps": total_steps, "parameter_count": total_params}
    print(f"[image_gen] Training complete: {result}")
    return result


# ---------------------------------------------------------------------------
# Legacy: Audio Encoder Training (CTC for STT) — kept for backward compat
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

            log_probs = model.forward_ctc(mel)
            log_probs = log_probs.permute(1, 0, 2)

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
# Legacy: Speech Decoder Training (codec) — kept for backward compat
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
            mel_hat, token_ids, commit_loss = model.forward_codec(mel)
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
