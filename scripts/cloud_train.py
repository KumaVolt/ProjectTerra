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


def upload_to_hf(result: dict, mode: str):
    """Upload checkpoint to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if not (hf_token and hf_repo):
        print("WARNING: HF_TOKEN/HF_REPO_ID not set. Checkpoint NOT uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, private=True, exist_ok=True)

    # Determine upload directory and HF path
    if mode == "multimodal":
        upload_dir = "models/current_multimodal"
        if not os.path.exists(upload_dir):
            upload_dir = "models/checkpoints/multimodal/best"
        hf_path = "multimodal-latest"
    elif mode == "sft":
        upload_dir = "models/checkpoints/sft/best"
        if not os.path.exists(upload_dir):
            upload_dir = "models/checkpoints/sft/final"
        hf_path = "sft-latest"
    else:
        upload_dir = "models/checkpoints/pretrain/best"
        if not os.path.exists(upload_dir):
            upload_dir = "models/checkpoints/pretrain/final"
        hf_path = "latest"

    print(f"Uploading {mode} checkpoint to HuggingFace: {hf_repo}/{hf_path}...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=hf_repo,
        path_in_repo=hf_path,
        commit_message=f"{mode}: {result.get('total_steps', 0)} steps, "
                      f"loss={result.get('best_val_loss', 'N/A')}, "
                      f"stop={result.get('stop_reason', 'unknown')}",
    )

    # Save and upload training result
    result_file = "training_done.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    api.upload_file(
        path_or_fileobj=result_file,
        path_in_repo=f"{hf_path}/training_result.json",
        repo_id=hf_repo,
    )
    print("Upload complete!")


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


def run_multimodal(max_steps: int) -> dict:
    """Run the full multimodal pipeline: tokenizers + unified fine-tuning."""
    # 1. Get base text model
    base_model_path = _find_or_download_base_model()

    # 2. Download multimodal data
    print("Downloading multimodal training data...")
    subprocess.run([sys.executable, "-c",
        "from src.data.multimodal_downloader import download_all_multimodal_data; download_all_multimodal_data(minimal=False)"],
        check=True)

    # 3. Train image tokenizer
    print("Training image tokenizer...")
    from src.training.train_multimodal import train_image_tokenizer
    img_result = train_image_tokenizer(
        data_dir="data/vision",
        output_dir="models/checkpoints/image_tokenizer",
        batch_size=32,
        max_steps=5000,
    )
    print(f"Image tokenizer done: {img_result.get('total_steps')} steps")

    # 4. Train audio tokenizer
    print("Training audio tokenizer...")
    from src.training.train_multimodal import train_audio_tokenizer
    audio_result = train_audio_tokenizer(
        data_dir="data/tts",
        output_dir="models/checkpoints/audio_tokenizer",
        batch_size=16,
        max_steps=3000,
    )
    print(f"Audio tokenizer done: {audio_result.get('total_steps')} steps")

    # 5. Multimodal fine-tuning
    mm_steps = max_steps if max_steps > 0 else 10000
    print(f"Starting multimodal fine-tuning ({mm_steps} steps)...")
    from src.training.train_multimodal import train_multimodal
    result = train_multimodal(
        text_model_path=base_model_path,
        image_tokenizer_path="models/checkpoints/image_tokenizer/best",
        audio_tokenizer_path="models/checkpoints/audio_tokenizer/best",
        vision_dir="data/vision",
        audio_dir="data/tts",
        output_dir="models/checkpoints/multimodal",
        batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        max_steps=mm_steps,
    )
    return result


def main():
    mode = os.environ.get("TRAINING_MODE", "pretrain")
    max_steps = int(os.environ.get("TRAINING_MAX_STEPS", "0"))
    max_samples = int(os.environ.get("SFT_MAX_SAMPLES", "20000"))

    setup_repo()

    if mode == "sft":
        result = run_sft(max_steps, max_samples)
    elif mode == "multimodal":
        result = run_multimodal(max_steps)
    else:
        result = run_pretrain(max_steps)

    print(f"Training complete: {result}")

    upload_to_hf(result, mode)
    self_destruct()


if __name__ == "__main__":
    main()
