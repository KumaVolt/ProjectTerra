#!/usr/bin/env python3
"""Standalone training script for cloud GPU pods.

This script runs on a RunPod/cloud GPU. It:
1. Clones the repo (or uses pre-uploaded code)
2. Runs pre-training OR SFT (based on TRAINING_MODE env var)
3. Uploads checkpoint to HuggingFace Hub
4. Self-destructs the pod

Environment variables (set by the pod):
    REPO_URL - Git repo to clone
    TRAINING_MODE - "pretrain", "sft", or "multimodal" (default: pretrain)
    TRAINING_MAX_STEPS - Max steps (0 = auto)
    SFT_MAX_SAMPLES - Max SFT training samples (default: 20000)
    HF_TOKEN - HuggingFace token for upload
    HF_REPO_ID - HuggingFace repo for checkpoints
    RUNPOD_API_KEY - For self-destruct
    RUNPOD_POD_ID - Auto-set by RunPod
"""

import json
import os
import subprocess
import sys


def setup_repo():
    """Clone repo and install dependencies."""
    repo_url = os.environ.get("REPO_URL", "")

    if repo_url and not os.path.exists("/workspace/terra/src"):
        print(f"Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, "/workspace/terra"], check=True)

    os.chdir("/workspace/terra")
    sys.path.insert(0, "/workspace/terra")

    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub", "--quiet"], check=True)


def ensure_pretrain_data():
    """Download and prepare pre-training data if not present."""
    if os.path.exists("data/pretrain_chunks/train.jsonl"):
        return

    print("No training data found. Initializing pipeline...")

    if not os.path.exists("models/tokenizer/tokenizer.json"):
        print("Training tokenizer...")
        subprocess.run([sys.executable, "-c",
            "from src.training.tokenizer import train_from_datasets; train_from_datasets(num_samples=100000)"],
            check=True)

    print("Downloading pre-training data...")
    subprocess.run([sys.executable, "-c",
        "from src.data.downloader import download_pretraining_data; download_pretraining_data(max_samples_per_source=50000)"],
        check=True)

    print("Preparing training chunks...")
    subprocess.run([sys.executable, "-c",
        "from src.data.downloader import prepare_pretraining_chunks; prepare_pretraining_chunks()"],
        check=True)


def run_pretrain(max_steps: int) -> dict:
    """Run pre-training from scratch."""
    ensure_pretrain_data()

    print(f"Starting pre-training (max_steps={max_steps})...")
    import yaml
    config = yaml.safe_load(open("configs/terra.yaml"))

    from src.training.pretrain import pretrain
    return pretrain(
        model_config=config.get("architecture", {}),
        data_path="data/pretrain_chunks",
        output_dir="models/checkpoints/pretrain",
        batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        max_steps=max_steps,
        warmup_steps=500,
        save_steps=0,
        eval_steps=0,
        patience=5,
        use_gradient_checkpointing=False,
    )


def run_sft(max_steps: int, max_samples: int) -> dict:
    """Run SFT on a pre-trained model."""
    # First, get the base model — either from HF or run pre-training
    base_model_path = _find_or_download_base_model()

    # Download SFT data
    print(f"Downloading SFT data (up to {max_samples} samples)...")
    from src.training.sft import download_sft_data
    download_sft_data(output_dir="data/sft", max_samples=max_samples)

    # Ensure tokenizer exists
    if not os.path.exists("models/tokenizer/tokenizer.json"):
        print("Training tokenizer...")
        subprocess.run([sys.executable, "-c",
            "from src.training.tokenizer import train_from_datasets; train_from_datasets(num_samples=100000)"],
            check=True)

    # Run SFT
    print(f"Starting SFT (max_steps={max_steps}, base={base_model_path})...")
    from src.training.sft import finetune
    return finetune(
        model_path=base_model_path,
        data_path="data/sft",
        output_dir="models/checkpoints/sft",
        batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        max_steps=max_steps,
        num_epochs=3,
        max_length=1024,
        patience=3,
    )


def _find_or_download_base_model() -> str:
    """Find a pre-trained base model to fine-tune."""
    # Check local checkpoints first
    candidates = [
        "models/checkpoints/pretrain/best",
        "models/checkpoints/pretrain/final",
        "models/current",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "config.json")):
            print(f"Found base model at {c}")
            return c

    # Try downloading from HuggingFace
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_token and hf_repo:
        print(f"Downloading base model from HuggingFace: {hf_repo}/latest...")
        try:
            from huggingface_hub import hf_hub_download
            out = "models/checkpoints/pretrain/best"
            os.makedirs(out, exist_ok=True)
            for fname in ["config.json", "model.safetensors"]:
                hf_hub_download(
                    repo_id=hf_repo, filename=f"latest/{fname}",
                    token=hf_token, local_dir=out,
                )
            # Files download to out/latest/
            actual_path = os.path.join(out, "latest")
            if os.path.exists(os.path.join(actual_path, "config.json")):
                print(f"Base model downloaded to {actual_path}")
                return actual_path
        except Exception as e:
            print(f"Could not download base model: {e}")

    # Last resort: run pre-training first
    print("No base model found. Running pre-training first...")
    result = run_pretrain(max_steps=0)
    return result.get("model_path", "models/checkpoints/pretrain/final")


def _find_upload_dir(candidates: list[str]) -> str | None:
    """Find the first existing directory with model files."""
    for d in candidates:
        if os.path.isdir(d) and any(
            f.endswith((".json", ".safetensors", ".pt", ".bin"))
            for f in os.listdir(d)
        ):
            return d
    return None


def _upload_folder_with_retry(api, folder_path: str, repo_id: str, path_in_repo: str, commit_message: str, retries: int = 3):
    """Upload folder with retries on failure."""
    import time as _time
    for attempt in range(retries):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
            )
            return True
        except Exception as e:
            print(f"  Upload attempt {attempt + 1}/{retries} failed: {e}", flush=True)
            if attempt < retries - 1:
                _time.sleep(10 * (attempt + 1))
    return False


def upload_to_hf(result: dict, mode: str):
    """Upload checkpoint to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if not (hf_token and hf_repo):
        print("WARNING: HF_TOKEN/HF_REPO_ID not set. Checkpoint NOT uploaded.", flush=True)
        return

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, private=True, exist_ok=True)

    # For multimodal mode, upload each component separately so partial progress is saved
    if mode == "multimodal":
        # Upload image tokenizer
        img_dir = _find_upload_dir([
            "models/checkpoints/image_tokenizer/best",
            "models/checkpoints/image_tokenizer/final",
            "models/checkpoints/image_tokenizer",
        ])
        if img_dir:
            print(f"Uploading image tokenizer from {img_dir}...", flush=True)
            _upload_folder_with_retry(api, img_dir, hf_repo, "image-tokenizer",
                                      f"image tokenizer checkpoint")

        # Upload audio tokenizer
        audio_dir = _find_upload_dir([
            "models/checkpoints/audio_tokenizer/best",
            "models/checkpoints/audio_tokenizer/final",
            "models/checkpoints/audio_tokenizer",
        ])
        if audio_dir:
            print(f"Uploading audio tokenizer from {audio_dir}...", flush=True)
            _upload_folder_with_retry(api, audio_dir, hf_repo, "audio-tokenizer",
                                      f"audio tokenizer checkpoint")

        # Upload multimodal model
        mm_dir = _find_upload_dir([
            "models/current_multimodal",
            "models/checkpoints/multimodal/best",
            "models/checkpoints/multimodal/final",
            "models/checkpoints/multimodal",
        ])
        if mm_dir:
            print(f"Uploading multimodal model from {mm_dir}...", flush=True)
            _upload_folder_with_retry(api, mm_dir, hf_repo, "multimodal-latest",
                                      f"multimodal: {result.get('total_steps', 0)} steps, "
                                      f"loss={result.get('best_val_loss', 'N/A')}")
        hf_path = "multimodal-latest"

    elif mode == "sft":
        upload_dir = _find_upload_dir([
            "models/checkpoints/sft/best",
            "models/checkpoints/sft/final",
        ])
        hf_path = "sft-latest"
        if upload_dir:
            print(f"Uploading SFT checkpoint from {upload_dir}...", flush=True)
            _upload_folder_with_retry(api, upload_dir, hf_repo, hf_path,
                                      f"sft: {result.get('total_steps', 0)} steps, "
                                      f"loss={result.get('best_val_loss', 'N/A')}")
    else:
        upload_dir = _find_upload_dir([
            "models/checkpoints/pretrain/best",
            "models/checkpoints/pretrain/final",
        ])
        hf_path = "latest"
        if upload_dir:
            print(f"Uploading pretrain checkpoint from {upload_dir}...", flush=True)
            _upload_folder_with_retry(api, upload_dir, hf_repo, hf_path,
                                      f"pretrain: {result.get('total_steps', 0)} steps, "
                                      f"loss={result.get('best_val_loss', 'N/A')}")

    # Save and upload training result
    result_file = "training_done.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    try:
        api.upload_file(
            path_or_fileobj=result_file,
            path_in_repo=f"{hf_path}/training_result.json",
            repo_id=hf_repo,
        )
    except Exception as e:
        print(f"WARNING: Could not upload training result: {e}", flush=True)
    print("Upload complete!", flush=True)


def self_destruct():
    """Terminate the RunPod pod."""
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not (pod_id and api_key):
        return

    print(f"Self-destructing pod {pod_id}...")
    import httpx
    try:
        httpx.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'},
            timeout=30,
        )
        print("Pod terminated.")
    except Exception as e:
        print(f"Could not self-destruct: {e}")


def _incremental_upload(local_dir: str, hf_path: str, label: str):
    """Upload a checkpoint to HF immediately (best-effort, non-blocking)."""
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if not (hf_token and hf_repo):
        return
    d = _find_upload_dir([local_dir, f"{local_dir}/best", f"{local_dir}/final"])
    if not d:
        print(f"[upload] No files found in {local_dir}, skipping {label} upload", flush=True)
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(hf_repo, private=True, exist_ok=True)
        print(f"[upload] Uploading {label} from {d} -> {hf_repo}/{hf_path}...", flush=True)
        _upload_folder_with_retry(api, d, hf_repo, hf_path, f"{label} checkpoint")
        print(f"[upload] {label} uploaded successfully!", flush=True)
    except Exception as e:
        print(f"[upload] WARNING: {label} upload failed: {e}", flush=True)


def run_multimodal(max_steps: int) -> dict:
    """Run the full multimodal pipeline: tokenizers + unified fine-tuning."""
    import sys as _sys
    # Force unbuffered output so logs appear in RunPod dashboard
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    # 1. Get base text model
    print("[multimodal] Step 1/5: Getting base text model...", flush=True)
    base_model_path = _find_or_download_base_model()
    print(f"[multimodal] Base model: {base_model_path}", flush=True)

    # 2. Download multimodal data — each modality separately so one failure doesn't kill all
    print("[multimodal] Step 2/5: Downloading multimodal data...", flush=True)
    # Install FFmpeg + audio codec support (torchcodec needs libavutil)
    subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "pip", "install", "torchcodec", "soundfile", "--quiet"])

    has_vision = False
    has_audio = False
    for modality, code, flag_name in [
        ("vision", "from src.data.multimodal_downloader import download_vision_data; download_vision_data(max_samples=1000, minimal=True)", "vision"),
        ("audio/tts", "from src.data.multimodal_downloader import download_tts_data; download_tts_data(max_samples=2000, minimal=True)", "audio"),
    ]:
        print(f"[multimodal]   Downloading {modality}...", flush=True)
        result = subprocess.run([sys.executable, "-c", code])
        if result.returncode != 0:
            print(f"[multimodal]   WARNING: {modality} download failed (code {result.returncode}), continuing...", flush=True)
        elif flag_name == "vision":
            has_vision = True
        elif flag_name == "audio":
            has_audio = True

    # Check if data actually exists on disk (download may "succeed" but produce nothing)
    has_vision = has_vision or os.path.isdir("data/vision")
    has_audio = has_audio or os.path.exists("data/tts/manifest.jsonl")
    print(f"[multimodal] Data: vision={has_vision}, audio={has_audio}", flush=True)

    # 3. Train image tokenizer
    img_tok_path = None
    if has_vision:
        print("[multimodal] Step 3/5: Training image tokenizer...", flush=True)
        from src.training.train_multimodal import train_image_tokenizer
        img_result = train_image_tokenizer(
            data_dir="data/vision",
            output_dir="models/checkpoints/image_tokenizer",
            batch_size=32,
            max_steps=5000,
        )
        print(f"[multimodal] Image tokenizer done: {img_result.get('total_steps')} steps", flush=True)
        _incremental_upload("models/checkpoints/image_tokenizer", "image-tokenizer", "image tokenizer")
        # Resolve actual path (best or final)
        for p in ["models/checkpoints/image_tokenizer/best", "models/checkpoints/image_tokenizer/final"]:
            if os.path.isdir(p):
                img_tok_path = p
                break
    else:
        print("[multimodal] Step 3/5: SKIPPED (no vision data)", flush=True)

    # 4. Train audio tokenizer
    audio_tok_path = None
    if has_audio:
        print("[multimodal] Step 4/5: Training audio tokenizer...", flush=True)
        from src.training.train_multimodal import train_audio_tokenizer
        audio_result = train_audio_tokenizer(
            data_dir="data/tts",
            output_dir="models/checkpoints/audio_tokenizer",
            batch_size=16,
            max_steps=3000,
        )
        print(f"[multimodal] Audio tokenizer done: {audio_result.get('total_steps')} steps", flush=True)
        _incremental_upload("models/checkpoints/audio_tokenizer", "audio-tokenizer", "audio tokenizer")
        for p in ["models/checkpoints/audio_tokenizer/best", "models/checkpoints/audio_tokenizer/final"]:
            if os.path.isdir(p):
                audio_tok_path = p
                break
    else:
        print("[multimodal] Step 4/5: SKIPPED (no audio data)", flush=True)

    if not img_tok_path and not audio_tok_path:
        print("[multimodal] ERROR: No tokenizers trained, cannot proceed to fine-tuning", flush=True)
        return {"error": "no_tokenizers", "total_steps": 0}

    # 5. Multimodal fine-tuning
    mm_steps = max_steps if max_steps > 0 else 10000
    print(f"[multimodal] Step 5/5: Multimodal fine-tuning ({mm_steps} steps)...", flush=True)
    print(f"[multimodal]   image_tok={img_tok_path}, audio_tok={audio_tok_path}", flush=True)
    from src.training.train_multimodal import train_multimodal
    result = train_multimodal(
        text_model_path=base_model_path,
        image_tokenizer_path=img_tok_path,
        audio_tokenizer_path=audio_tok_path,
        vision_dir="data/vision" if has_vision else None,
        audio_dir="data/tts" if has_audio else None,
        output_dir="models/checkpoints/multimodal",
        batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        max_steps=mm_steps,
    )
    print(f"[multimodal] All done! Result: {result}", flush=True)
    return result


def main():
    import sys as _sys
    # Force unbuffered output so logs appear in RunPod dashboard immediately
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    mode = os.environ.get("TRAINING_MODE", "pretrain")
    max_steps = int(os.environ.get("TRAINING_MAX_STEPS", "0"))
    max_samples = int(os.environ.get("SFT_MAX_SAMPLES", "20000"))

    print(f"[cloud_train] Starting: mode={mode}, max_steps={max_steps}", flush=True)

    try:
        setup_repo()
        print(f"[cloud_train] Repo setup complete, starting {mode}...", flush=True)

        if mode == "sft":
            result = run_sft(max_steps, max_samples)
        elif mode == "multimodal":
            result = run_multimodal(max_steps)
        else:
            result = run_pretrain(max_steps)

        print(f"[cloud_train] Training complete: {result}", flush=True)
        upload_to_hf(result, mode)

    except Exception as e:
        print(f"[cloud_train] FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

    self_destruct()


if __name__ == "__main__":
    main()
