"""Download and prepare training data for all multimodal components.

Each modality needs different data:
- Vision encoder: image-text pairs (contrastive learning)
- Image generator: image-caption pairs (diffusion training)
- Audio encoder (STT): audio-transcript pairs (ASR)
- Speech decoder (TTS): audio-text pairs (speech synthesis)

All datasets stream from HuggingFace to avoid filling disk.
"""

import io
import json
import struct
import wave
from pathlib import Path

import torch
from datasets import load_dataset


# ── Vision Data (image-text pairs) ──

VISION_SOURCES = {
    "cc3m_conceptual_captions": {
        "hf_name": "pixparse/cc3m-wds",
        "split": "train",
        "image_field": "jpg",
        "caption_field": "txt",
        "description": "Conceptual Captions 3M — image-alt text pairs from the web",
    },
    "coco_captions": {
        "hf_name": "HuggingFaceM4/COCO",
        "split": "train",
        "image_field": "image",
        "caption_field": "sentences_raw",
        "description": "COCO Captions — 330K images with 5 human captions each",
    },
}

VISION_SOURCES_MINIMAL = {
    "coco_captions": VISION_SOURCES["coco_captions"],
}


def download_vision_data(
    output_dir: str = "data/vision",
    max_samples: int = 50000,
    image_size: int = 384,
    minimal: bool = False,
) -> dict:
    """Download image-caption pairs for vision encoder training.

    Saves images as tensors and captions as text.
    Used for: contrastive pre-training (SigLIP-style) of vision encoder,
    and image-text alignment fine-tuning.

    Returns: dict with paths and counts.
    """
    from torchvision import transforms

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    sources = VISION_SOURCES_MINIMAL if minimal else VISION_SOURCES
    total_saved = 0

    captions_file = out / "captions.jsonl"
    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)

    with open(captions_file, "w") as cap_f:
        for source_name, info in sources.items():
            print(f"[vision] Downloading {source_name}: {info['description']}...")

            try:
                ds = load_dataset(info["hf_name"], split=info["split"], streaming=True)
                count = 0

                for example in ds:
                    if count >= max_samples:
                        break

                    try:
                        image = _extract_image(example, info)
                        caption = _extract_caption(example, info)

                        if image is None or not caption:
                            continue

                        # Convert to RGB if needed
                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        # Transform and save
                        tensor = transform(image)
                        img_path = images_dir / f"{total_saved:07d}.pt"
                        torch.save(tensor, img_path)

                        cap_f.write(json.dumps({
                            "id": total_saved,
                            "image": str(img_path),
                            "caption": caption,
                            "source": source_name,
                        }) + "\n")

                        total_saved += 1
                        count += 1

                        if count % 5000 == 0:
                            print(f"  [{source_name}] {count}/{max_samples}")

                    except Exception:
                        continue

                print(f"[vision] {source_name}: saved {count} pairs")

            except Exception as e:
                print(f"[vision] {source_name} failed: {e}")

    print(f"[vision] Total: {total_saved} image-caption pairs saved to {out}")
    return {"total": total_saved, "captions": str(captions_file), "images": str(images_dir)}


def _extract_image(example: dict, info: dict):
    """Extract PIL image from dataset example."""
    from PIL import Image

    field = info["image_field"]
    img = example.get(field)
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img))
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"]))
    return None


def _extract_caption(example: dict, info: dict) -> str:
    """Extract caption text from dataset example."""
    field = info["caption_field"]
    caption = example.get(field, "")
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8", errors="ignore")
    return str(caption).strip()


# ── Image Generation Data ──

IMAGE_GEN_SOURCES = {
    "laion_aesthetics": {
        "hf_name": "laion/laion-art",
        "split": "train",
        "image_field": "image",
        "caption_field": "TEXT",
        "description": "LAION Aesthetics — high-quality aesthetic images with captions",
    },
    "diffusiondb": {
        "hf_name": "poloclub/diffusiondb",
        "hf_subset": "random_1k",
        "split": "train",
        "image_field": "image",
        "caption_field": "prompt",
        "description": "DiffusionDB — Stable Diffusion generations with prompts (learn prompt→image mapping)",
    },
}


def download_image_gen_data(
    output_dir: str = "data/image_gen",
    max_samples: int = 50000,
    image_size: int = 256,
    minimal: bool = False,
) -> dict:
    """Download image-caption pairs for diffusion model training.

    Same format as vision data but at the image generator's target resolution.
    """
    from torchvision import transforms

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1] for diffusion
    ])

    sources = {"diffusiondb": IMAGE_GEN_SOURCES["diffusiondb"]} if minimal else IMAGE_GEN_SOURCES
    total_saved = 0

    captions_file = out / "captions.jsonl"
    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)

    with open(captions_file, "w") as cap_f:
        for source_name, info in sources.items():
            print(f"[image_gen] Downloading {source_name}: {info['description']}...")

            try:
                kwargs = {"path": info["hf_name"], "split": info["split"], "streaming": True}
                if "hf_subset" in info:
                    kwargs["name"] = info["hf_subset"]

                ds = load_dataset(**kwargs)
                count = 0

                for example in ds:
                    if count >= max_samples:
                        break

                    try:
                        image = _extract_image(example, info)
                        caption = _extract_caption(example, info)

                        if image is None or not caption:
                            continue

                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        tensor = transform(image)
                        img_path = images_dir / f"{total_saved:07d}.pt"
                        torch.save(tensor, img_path)

                        cap_f.write(json.dumps({
                            "id": total_saved,
                            "image": str(img_path),
                            "caption": caption,
                            "source": source_name,
                        }) + "\n")

                        total_saved += 1
                        count += 1

                        if count % 5000 == 0:
                            print(f"  [{source_name}] {count}/{max_samples}")

                    except Exception:
                        continue

                print(f"[image_gen] {source_name}: saved {count} pairs")

            except Exception as e:
                print(f"[image_gen] {source_name} failed: {e}")

    print(f"[image_gen] Total: {total_saved} image-caption pairs saved to {out}")
    return {"total": total_saved, "captions": str(captions_file), "images": str(images_dir)}


# ── Audio / STT Data ──

AUDIO_SOURCES = {
    "librispeech_clean": {
        "hf_name": "librispeech_asr",
        "hf_subset": "clean",
        "split": "train.100",
        "audio_field": "audio",
        "text_field": "text",
        "description": "LibriSpeech clean — 100h read English speech",
    },
    "common_voice": {
        "hf_name": "mozilla-foundation/common_voice_17_0",
        "hf_subset": "en",
        "split": "train",
        "audio_field": "audio",
        "text_field": "sentence",
        "description": "Common Voice — crowd-sourced multilingual speech",
    },
}


def download_audio_data(
    output_dir: str = "data/audio",
    max_samples: int = 20000,
    target_sr: int = 16000,
    minimal: bool = False,
) -> dict:
    """Download audio-transcript pairs for audio encoder (STT) training.

    Saves mel spectrograms (pre-computed for efficient training) + transcripts.

    Returns: dict with paths and counts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = {"librispeech_clean": AUDIO_SOURCES["librispeech_clean"]} if minimal else AUDIO_SOURCES
    total_saved = 0

    manifest_file = out / "manifest.jsonl"
    mels_dir = out / "mels"
    mels_dir.mkdir(exist_ok=True)

    with open(manifest_file, "w") as manifest:
        for source_name, info in sources.items():
            print(f"[audio] Downloading {source_name}: {info['description']}...")

            try:
                kwargs = {"path": info["hf_name"], "split": info["split"], "streaming": True}
                if "hf_subset" in info:
                    kwargs["name"] = info["hf_subset"]

                ds = load_dataset(**kwargs)
                count = 0

                for example in ds:
                    if count >= max_samples:
                        break

                    try:
                        audio_data = example.get(info["audio_field"])
                        text = example.get(info["text_field"], "")

                        if audio_data is None or not text:
                            continue

                        # Extract audio array and sample rate
                        if isinstance(audio_data, dict):
                            array = audio_data.get("array")
                            sr = audio_data.get("sampling_rate", target_sr)
                        else:
                            continue

                        if array is None or len(array) < 1600:  # < 0.1s
                            continue

                        audio_tensor = torch.tensor(array, dtype=torch.float32)

                        # Resample if needed
                        if sr != target_sr:
                            import torchaudio
                            audio_tensor = torchaudio.functional.resample(
                                audio_tensor.unsqueeze(0), sr, target_sr
                            ).squeeze(0)

                        # Compute mel spectrogram
                        mel = _compute_mel(audio_tensor, target_sr)

                        # Save
                        mel_path = mels_dir / f"{total_saved:07d}.pt"
                        torch.save(mel, mel_path)

                        manifest.write(json.dumps({
                            "id": total_saved,
                            "mel": str(mel_path),
                            "text": text.strip(),
                            "duration_sec": len(array) / sr,
                            "source": source_name,
                        }) + "\n")

                        total_saved += 1
                        count += 1

                        if count % 2000 == 0:
                            print(f"  [{source_name}] {count}/{max_samples}")

                    except Exception:
                        continue

                print(f"[audio] {source_name}: saved {count} samples")

            except Exception as e:
                print(f"[audio] {source_name} failed: {e}")

    total_hours = _count_hours(manifest_file)
    print(f"[audio] Total: {total_saved} samples ({total_hours:.1f}h) saved to {out}")
    return {"total": total_saved, "hours": total_hours, "manifest": str(manifest_file), "mels": str(mels_dir)}


def _compute_mel(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> torch.Tensor:
    """Compute mel spectrogram from raw audio waveform.

    Returns: (n_mels, time_frames) tensor.
    """
    # Use torchaudio if available, otherwise manual
    try:
        import torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel = mel_transform(audio.unsqueeze(0)).squeeze(0)
    except ImportError:
        # Manual mel spectrogram via torch STFT
        window = torch.hann_window(n_fft)
        stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        power = stft.abs().pow(2)
        # Simple mel filterbank (approximate)
        n_freqs = power.shape[0]
        mel_fb = torch.linspace(0, sample_rate // 2, n_mels + 2)
        mel_filter = torch.zeros(n_mels, n_freqs)
        freqs = torch.linspace(0, sample_rate // 2, n_freqs)
        for i in range(n_mels):
            low, center, high = mel_fb[i], mel_fb[i + 1], mel_fb[i + 2]
            mel_filter[i] = ((freqs >= low) & (freqs <= high)).float()
        mel = mel_filter @ power

    # Log-mel
    mel = torch.log(mel.clamp(min=1e-10))
    return mel


def _count_hours(manifest_path: Path) -> float:
    """Sum duration from manifest file."""
    total = 0.0
    try:
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                total += entry.get("duration_sec", 0)
    except Exception:
        pass
    return total / 3600.0


# ── TTS Data (same audio but optimized for speech synthesis) ──

TTS_SOURCES = {
    "ljspeech": {
        "hf_name": "keithito/lj_speech",
        "split": "train",
        "audio_field": "audio",
        "text_field": "normalized_text",
        "description": "LJ Speech — 24h single-speaker English (clean, ideal for TTS)",
    },
    "libritts_clean": {
        "hf_name": "cdminix/libritts-r-aligned",
        "hf_subset": "clean",
        "split": "train.clean.100",
        "audio_field": "audio",
        "text_field": "text_normalized",
        "description": "LibriTTS-R clean — multi-speaker English speech at 24kHz",
    },
}


def download_tts_data(
    output_dir: str = "data/tts",
    max_samples: int = 20000,
    target_sr: int = 24000,
    minimal: bool = False,
) -> dict:
    """Download audio-text pairs for speech decoder (TTS) codec training.

    Higher sample rate (24kHz) than STT because we need to reconstruct audio quality.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = {"ljspeech": TTS_SOURCES["ljspeech"]} if minimal else TTS_SOURCES
    total_saved = 0

    manifest_file = out / "manifest.jsonl"
    mels_dir = out / "mels"
    mels_dir.mkdir(exist_ok=True)

    with open(manifest_file, "w") as manifest:
        for source_name, info in sources.items():
            print(f"[tts] Downloading {source_name}: {info['description']}...")

            try:
                kwargs = {"path": info["hf_name"], "split": info["split"], "streaming": True}
                if "hf_subset" in info:
                    kwargs["name"] = info["hf_subset"]

                ds = load_dataset(**kwargs)
                count = 0

                for example in ds:
                    if count >= max_samples:
                        break

                    try:
                        audio_data = example.get(info["audio_field"])
                        text = example.get(info["text_field"], "")

                        if audio_data is None or not text:
                            continue

                        if isinstance(audio_data, dict):
                            array = audio_data.get("array")
                            sr = audio_data.get("sampling_rate", target_sr)
                        else:
                            continue

                        if array is None or len(array) < 4800:  # < 0.2s
                            continue

                        audio_tensor = torch.tensor(array, dtype=torch.float32)

                        if sr != target_sr:
                            import torchaudio
                            audio_tensor = torchaudio.functional.resample(
                                audio_tensor.unsqueeze(0), sr, target_sr
                            ).squeeze(0)

                        mel = _compute_mel(audio_tensor, target_sr, n_fft=1024, hop_length=256, n_mels=80)

                        mel_path = mels_dir / f"{total_saved:07d}.pt"
                        torch.save(mel, mel_path)

                        manifest.write(json.dumps({
                            "id": total_saved,
                            "mel": str(mel_path),
                            "text": text.strip(),
                            "duration_sec": len(array) / sr,
                            "source": source_name,
                        }) + "\n")

                        total_saved += 1
                        count += 1

                        if count % 2000 == 0:
                            print(f"  [{source_name}] {count}/{max_samples}")

                    except Exception:
                        continue

                print(f"[tts] {source_name}: saved {count} samples")

            except Exception as e:
                print(f"[tts] {source_name} failed: {e}")

    total_hours = _count_hours(manifest_file)
    print(f"[tts] Total: {total_saved} samples ({total_hours:.1f}h) saved to {out}")
    return {"total": total_saved, "hours": total_hours, "manifest": str(manifest_file)}


# ── Dataset classes for training ──

class VisionDataset(torch.utils.data.Dataset):
    """Load vision data for training the vision encoder or image generator."""

    def __init__(self, data_dir: str, tokenizer_path: str = "models/tokenizer", max_caption_len: int = 77):
        self.data_dir = Path(data_dir)
        self.captions_file = self.data_dir / "captions.jsonl"
        self.max_caption_len = max_caption_len

        # Load manifest
        self.entries = []
        with open(self.captions_file) as f:
            for line in f:
                self.entries.append(json.loads(line))

        # Load tokenizer
        from src.training.tokenizer import load_tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        image = torch.load(entry["image"], weights_only=True)
        caption = entry["caption"]

        # Tokenize caption
        encoded = self.tokenizer.encode(caption)
        token_ids = encoded.ids[:self.max_caption_len]
        # Pad
        token_ids = token_ids + [0] * (self.max_caption_len - len(token_ids))

        return {
            "pixel_values": image,
            "caption_ids": torch.tensor(token_ids, dtype=torch.long),
            "caption": caption,
        }


class AudioDataset(torch.utils.data.Dataset):
    """Load audio data for training the audio encoder (STT) or speech decoder (TTS)."""

    def __init__(self, data_dir: str, tokenizer_path: str = "models/tokenizer",
                 max_mel_len: int = 3000, max_text_len: int = 256):
        self.data_dir = Path(data_dir)
        self.manifest_file = self.data_dir / "manifest.jsonl"
        self.max_mel_len = max_mel_len
        self.max_text_len = max_text_len

        self.entries = []
        with open(self.manifest_file) as f:
            for line in f:
                self.entries.append(json.loads(line))

        from src.training.tokenizer import load_tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        mel = torch.load(entry["mel"], weights_only=True)

        # Pad or truncate mel
        if mel.shape[1] > self.max_mel_len:
            mel = mel[:, :self.max_mel_len]
        elif mel.shape[1] < self.max_mel_len:
            pad = torch.zeros(mel.shape[0], self.max_mel_len - mel.shape[1])
            mel = torch.cat([mel, pad], dim=1)

        # Tokenize text
        encoded = self.tokenizer.encode(entry["text"])
        token_ids = encoded.ids[:self.max_text_len]
        text_len = len(token_ids)
        token_ids = token_ids + [0] * (self.max_text_len - len(token_ids))

        return {
            "mel": mel,
            "text_ids": torch.tensor(token_ids, dtype=torch.long),
            "text_len": text_len,
            "text": entry["text"],
        }


# ── Convenience: download everything ──

def download_all_multimodal_data(minimal: bool = True) -> dict:
    """Download training data for all modalities.

    Args:
        minimal: If True, download small samples for testing.
                 If False, download full training sets.

    Returns: dict of results per modality.
    """
    max_vision = 5000 if minimal else 50000
    max_audio = 2000 if minimal else 20000
    max_tts = 2000 if minimal else 20000
    max_image_gen = 1000 if minimal else 50000

    results = {}

    print("=" * 60)
    print("Downloading multimodal training data")
    print("=" * 60)

    print("\n[1/4] Vision data (image-caption pairs)...")
    results["vision"] = download_vision_data(max_samples=max_vision, minimal=minimal)

    print("\n[2/4] Image generation data...")
    results["image_gen"] = download_image_gen_data(max_samples=max_image_gen, minimal=minimal)

    print("\n[3/4] Audio/STT data (speech-transcript pairs)...")
    results["audio"] = download_audio_data(max_samples=max_audio, minimal=minimal)

    print("\n[4/4] TTS data (high-quality speech)...")
    results["tts"] = download_tts_data(max_samples=max_tts, minimal=minimal)

    print("\n" + "=" * 60)
    print("Summary:")
    for name, r in results.items():
        print(f"  {name}: {r.get('total', '?')} samples")
    print("=" * 60)

    return results
