#!/usr/bin/env python3
"""Standalone training script for cloud GPU pods.

This script runs on a RunPod/cloud GPU. It:
1. Clones the repo (or uses pre-uploaded code)
2. Runs pre-training with auto steps
3. Uploads checkpoint to HuggingFace Hub
4. Self-destructs the pod

Environment variables (set by the pod):
    REPO_URL - Git repo to clone
    TRAINING_MAX_STEPS - Max steps (0 = auto)
    HF_TOKEN - HuggingFace token for upload
    HF_REPO_ID - HuggingFace repo for checkpoints
    RUNPOD_API_KEY - For self-destruct
    RUNPOD_POD_ID - Auto-set by RunPod
"""

import json
import os
import subprocess
import sys


def main():
    repo_url = os.environ.get("REPO_URL", "")
    max_steps = int(os.environ.get("TRAINING_MAX_STEPS", "0"))

    # Clone repo if not already present
    if repo_url and not os.path.exists("/workspace/terra/src"):
        print(f"Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, "/workspace/terra"], check=True)

    os.chdir("/workspace/terra")
    sys.path.insert(0, "/workspace/terra")

    # Install
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub", "--quiet"], check=True)

    # Download and prepare data if not present
    if not os.path.exists("data/pretrain_chunks/train.jsonl"):
        print("No training data found. Initializing pipeline...")
        # Train tokenizer if needed
        if not os.path.exists("models/tokenizer/tokenizer.json"):
            print("Training tokenizer...")
            subprocess.run([sys.executable, "-c",
                "from src.training.tokenizer import train_from_datasets; train_from_datasets(num_samples=100000)"],
                check=True)
        # Download data
        print("Downloading pre-training data...")
        subprocess.run([sys.executable, "-c",
            "from src.data.downloader import download_pretraining_data; download_pretraining_data(max_samples_per_source=50000)"],
            check=True)
        # Prepare chunks
        print("Preparing training chunks...")
        subprocess.run([sys.executable, "-c",
            "from src.data.downloader import prepare_pretraining_chunks; prepare_pretraining_chunks()"],
            check=True)

    # Run pre-training
    print(f"Starting pre-training (max_steps={max_steps})...")
    import yaml
    config = yaml.safe_load(open("configs/terra.yaml"))

    from src.training.pretrain import pretrain
    result = pretrain(
        model_config=config.get("architecture", {}),
        data_path="data/pretrain_chunks",
        output_dir="models/checkpoints/pretrain",
        batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        max_steps=max_steps,
        warmup_steps=500,
        save_steps=0,  # auto
        eval_steps=0,  # auto
        patience=5,
        use_gradient_checkpointing=False,
    )

    # Save result
    with open("training_done.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Training complete: {result}")

    # Upload to HuggingFace
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_token and hf_repo:
        print(f"Uploading to HuggingFace: {hf_repo}")
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(hf_repo, private=True, exist_ok=True)

        # Upload best model if available, otherwise final
        upload_dir = "models/checkpoints/pretrain/best"
        if not os.path.exists(upload_dir):
            upload_dir = "models/checkpoints/pretrain/final"

        api.upload_folder(
            folder_path=upload_dir,
            repo_id=hf_repo,
            path_in_repo="latest",
            commit_message=f"Training: {result.get('total_steps', 0)} steps, "
                          f"loss={result.get('best_val_loss', 'N/A')}, "
                          f"stop={result.get('stop_reason', 'unknown')}",
        )
        api.upload_file(
            path_or_fileobj="training_done.json",
            path_in_repo="latest/training_result.json",
            repo_id=hf_repo,
        )
        print("Upload complete!")
    else:
        print("WARNING: HF_TOKEN/HF_REPO_ID not set. Checkpoint NOT uploaded.")

    # Self-destruct
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if pod_id and api_key:
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


if __name__ == "__main__":
    main()
