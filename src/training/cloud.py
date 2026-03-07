"""Cloud GPU burst training - dynamically create and destroy pods.

Supports:
- Modal (serverless, auto-shutdown, pay-per-second)
- RunPod (raw GPU pods, cheaper for long runs)

Two modes:
- Synchronous: terra cloud-train (blocks until done - need PC on)
- Async:       terra cloud-train --async (fire-and-forget, close laptop)
               terra cloud-status          (check progress)
               terra cloud-download         (pull results when done)

The pod is created on demand, runs training, and shuts down automatically.
"""

import json
import os
import subprocess
import time
from pathlib import Path

# File that tracks active cloud jobs
CLOUD_STATE_FILE = Path("configs/cloud_job.json")


def cloud_pretrain(
    config_path: str = "configs/terra.yaml",
    provider: str | None = None,
    gpu: str = "RTX5090",
    max_steps: int = 10000,
    sync_checkpoint: bool = True,
    async_mode: bool = False,
    gpu_count: int = 1,
    data_scale: str = "small",
    model_preset: str = "",
    network_volume_id: str = "",
) -> dict:
    """Run pre-training on a cloud GPU pod.

    Args:
        config_path: Path to terra config.
        provider: "modal" or "runpod". Auto-detects if None.
        gpu: GPU type ("A100", "H100", "A10G", "L4").
        max_steps: Training steps to run.
        sync_checkpoint: Download checkpoint after training.
        async_mode: If True, start job and return immediately (fire-and-forget).

    Returns:
        Training result dict. In async mode, returns job info instead.
    """
    if provider is None:
        provider = _detect_provider()

    extra_env = [
        {"key": "TRAINING_DATA_SCALE", "value": data_scale},
        {"key": "TRAINING_MODEL_PRESET", "value": model_preset},
    ]

    if provider == "modal":
        return _modal_pretrain(config_path, gpu, max_steps, sync_checkpoint, async_mode)
    elif provider == "runpod":
        return _runpod_pretrain(config_path, gpu, max_steps, sync_checkpoint, async_mode, gpu_count=gpu_count, extra_env=extra_env, network_volume_id=network_volume_id)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'modal' or 'runpod'.")


def cloud_status() -> dict:
    """Check the status of an active cloud training job."""
    if not CLOUD_STATE_FILE.exists():
        return {"status": "no_active_job"}

    state = json.loads(CLOUD_STATE_FILE.read_text())
    provider = state.get("provider")

    if provider == "runpod":
        return _runpod_check_status(state)
    elif provider == "modal":
        return _modal_check_status(state)
    else:
        return {"status": "unknown_provider", **state}


def cloud_download(output_dir: str = "models/checkpoints/pretrain/final") -> dict:
    """Download results from a completed cloud training job and destroy the pod."""
    if not CLOUD_STATE_FILE.exists():
        return {"error": "No active cloud job found"}

    state = json.loads(CLOUD_STATE_FILE.read_text())
    provider = state.get("provider")

    if provider == "runpod":
        result = _runpod_download_and_destroy(state, output_dir)
    elif provider == "modal":
        result = _modal_download(state, output_dir)
    else:
        return {"error": f"Unknown provider: {provider}"}

    # Clean up job state
    CLOUD_STATE_FILE.unlink(missing_ok=True)
    return result


def _detect_provider() -> str:
    """Auto-detect which cloud provider is configured."""
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return "modal"
    if os.environ.get("RUNPOD_API_KEY"):
        return "runpod"
    raise RuntimeError(
        "No cloud GPU provider configured. Set either:\n"
        "  - MODAL_TOKEN_ID + MODAL_TOKEN_SECRET (for Modal)\n"
        "  - RUNPOD_API_KEY (for RunPod)"
    )


def _save_cloud_state(state: dict):
    """Save cloud job state to disk for async tracking."""
    CLOUD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLOUD_STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# RunPod
# ---------------------------------------------------------------------------

# Training script that runs on the pod.
# After training: uploads checkpoint to HuggingFace Hub, then self-destructs.
# No need to keep the pod alive — results are safe on HF.
RUNPOD_TRAIN_CMD = """
cd /workspace &&
git clone {repo_url} terra 2>/dev/null || true &&
cd terra &&
pip install -e . --quiet &&
pip install huggingface-hub --quiet &&
python -c "
import yaml, json, os
config = yaml.safe_load(open('configs/terra.yaml'))
from src.training.pretrain import pretrain
result = pretrain(
    model_config=config.get('architecture', {{}}),
    data_path='data/pretrain_chunks',
    output_dir='models/checkpoints/pretrain',
    batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    max_steps={max_steps},  # 0 = auto based on data size
    warmup_steps=500,
    save_steps=1000,
    eval_steps=500,
    patience=5,
    use_gradient_checkpointing=False,
)
with open('/workspace/terra/training_done.json', 'w') as f:
    json.dump(result, f, indent=2)
print('TRAINING_COMPLETE')

# Upload checkpoint to HuggingFace Hub
hf_token = os.environ.get('HF_TOKEN', '')
hf_repo = os.environ.get('HF_REPO_ID', '')
if hf_token and hf_repo:
    print(f'Uploading checkpoint to HuggingFace: {{hf_repo}}')
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, private=True, exist_ok=True)
    # Upload best model (or final if no best)
    import os.path
    upload_dir = 'models/checkpoints/pretrain/best' if os.path.exists('models/checkpoints/pretrain/best') else 'models/checkpoints/pretrain/final'
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=hf_repo,
        path_in_repo='latest',
        commit_message=f'Training complete: {{result.get(\"total_steps\", 0)}} steps, loss={{result.get(\"best_val_loss\", \"N/A\")}}',
    )
    # Also upload training results
    api.upload_file(
        path_or_fileobj='/workspace/terra/training_done.json',
        path_in_repo='latest/training_result.json',
        repo_id=hf_repo,
    )
    print('Upload complete!')
else:
    print('WARNING: HF_TOKEN or HF_REPO_ID not set, checkpoint NOT uploaded.')
    print('The pod will self-destruct and results will be LOST.')

# Self-destruct: terminate the pod completely
pod_id = os.environ.get('RUNPOD_POD_ID', '')
api_key = os.environ.get('RUNPOD_API_KEY', '')
if pod_id and api_key:
    print(f'Self-destructing pod {{pod_id}}...')
    import httpx
    try:
        httpx.post(
            'https://api.runpod.io/graphql',
            headers={{'Authorization': f'Bearer {{api_key}}', 'Content-Type': 'application/json'}},
            json={{'query': 'mutation {{ podTerminate(input: {{podId: \\\"' + pod_id + '\\\"}}) }}'}},
            timeout=30,
        )
    except Exception as e:
        print(f'Could not self-destruct: {{e}}')
"
"""


def _runpod_client():
    """Create an httpx client for RunPod API."""
    import httpx
    api_key = os.environ["RUNPOD_API_KEY"]
    return httpx.Client(timeout=60), {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _runpod_graphql(client, headers, query: str, variables: dict = None) -> dict:
    """Execute a RunPod GraphQL query."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = client.post("https://api.runpod.io/graphql", headers=headers, json=payload)
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"RunPod API error: {data['errors']}")
    return data


def _runpod_launch_pod(
    gpu: str, max_steps: int, async_mode: bool = True,
    mode: str = "pretrain", extra_env: list = None,
    gpu_count: int = 1, network_volume_id: str = "",
) -> dict:
    """Launch a RunPod GPU pod for training (pretrain or SFT).

    This is the shared pod creation logic used by both cloud-train and cloud-sft.
    """
    client, headers = _runpod_client()

    gpu_map = {
        "A100": "NVIDIA A100 80GB PCIe",
        "A100-SXM": "NVIDIA A100-SXM4-80GB",
        "H100": "NVIDIA H100 80GB HBM3",
        "H200": "NVIDIA H200 NVL",
        "H200-SXM": "NVIDIA H200",
        "L4": "NVIDIA L4",
        "L40S": "NVIDIA L40S",
        "A40": "NVIDIA A40",
        "RTX5090": "NVIDIA GeForce RTX 5090",
        "RTX4090": "NVIDIA GeForce RTX 4090",
        "RTX3090": "NVIDIA GeForce RTX 3090",
    }
    gpu_id = gpu_map.get(gpu, gpu)

    try:
        repo_url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        repo_url = ""

    pod_name = f"terra-{mode}-{int(time.time())}"
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    rp_key = os.environ["RUNPOD_API_KEY"]

    print(f"[cloud/runpod] Creating {mode} pod with {gpu_id}...")

    query = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
        }
    }
    """
    docker_args = "bash -c 'cd /workspace && (git clone $REPO_URL terra 2>/dev/null || cd terra && git pull) && cd terra && python scripts/cloud_train.py'"

    env = [
        {"key": "TRAINING_MODE", "value": mode},
        {"key": "TRAINING_MAX_STEPS", "value": str(max_steps)},
        {"key": "TRAINING_GPU_COUNT", "value": str(gpu_count)},
        {"key": "RUNPOD_API_KEY", "value": rp_key},
        {"key": "HF_TOKEN", "value": hf_token},
        {"key": "HF_REPO_ID", "value": hf_repo},
        {"key": "REPO_URL", "value": repo_url},
    ]
    if extra_env:
        env.extend(extra_env)

    pod_input = {
        "name": pod_name,
        "imageName": "runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeId": gpu_id,
        "gpuCount": gpu_count,
        "volumeInGb": 0,
        "containerDiskInGb": 50 if network_volume_id else (200 if mode == "pretrain" else 50),
        "minVcpuCount": 1,
        "minMemoryInGb": 1,
        "dockerArgs": docker_args,
        "env": env,
    }
    if network_volume_id:
        pod_input["networkVolumeId"] = network_volume_id

    variables = {"input": pod_input}

    resp = client.post(
        "https://api.runpod.io/graphql",
        headers=headers,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    result = resp.json()

    if "errors" in result:
        raise RuntimeError(f"RunPod API error: {result['errors']}")

    pod_data = result["data"]["podFindAndDeployOnDemand"]
    pod_id = pod_data["id"]
    print(f"[cloud/runpod] Pod created: {pod_id}")

    job_state = {
        "provider": "runpod",
        "pod_id": pod_id,
        "gpu": gpu,
        "mode": mode,
        "max_steps": max_steps,
        "started_at": time.time(),
        "status": "running",
    }
    _save_cloud_state(job_state)

    if async_mode:
        print(f"[cloud/runpod] {mode} job running in background. Close your laptop safely.")
        print(f"[cloud/runpod] Check progress: terra cloud-status")
        print(f"[cloud/runpod] Download results: terra cloud-download")
        client.close()
        return {"status": "started", "pod_id": pod_id, "provider": "runpod", "mode": mode}

    client.close()
    return {"status": "started", "pod_id": pod_id, "provider": "runpod", "mode": mode}


def _runpod_pretrain(
    config_path: str, gpu: str, max_steps: int, sync_checkpoint: bool, async_mode: bool,
    gpu_count: int = 1, extra_env: list = None, network_volume_id: str = "",
) -> dict:
    """Run pre-training on a RunPod GPU pod."""
    result = _runpod_launch_pod(gpu=gpu, max_steps=max_steps, async_mode=async_mode, mode="pretrain", gpu_count=gpu_count, extra_env=extra_env, network_volume_id=network_volume_id)

    if async_mode:
        return result

    pod_id = result["pod_id"]
    client, headers = _runpod_client()

    # Synchronous mode: wait for completion
    try:
        print("[cloud/runpod] Waiting for training to complete...")
        _runpod_wait_for_completion(client, headers, pod_id)

        if sync_checkpoint:
            print("[cloud/runpod] Downloading checkpoint...")
            _runpod_download_results(client, headers, pod_id)
            return {"status": "completed", "model_path": "models/checkpoints/pretrain/final"}

        return {"status": "completed", "pod_id": pod_id}

    finally:
        print(f"[cloud/runpod] Destroying pod {pod_id}...")
        _runpod_destroy(client, headers, pod_id)
        CLOUD_STATE_FILE.unlink(missing_ok=True)
        client.close()
        print("[cloud/runpod] Pod destroyed.")


def _runpod_check_status(state: dict) -> dict:
    """Check status of a running RunPod job."""
    client, headers = _runpod_client()
    pod_id = state["pod_id"]

    try:
        result = _runpod_graphql(client, headers, """
            query pod($podId: String!) {
                pod(input: {podId: $podId}) {
                    id
                    desiredStatus
                    runtime {
                        uptimeInSeconds
                        gpus { gpuUtilPercent memoryUtilPercent }
                    }
                }
            }
        """, {"podId": pod_id})

        pod = result.get("data", {}).get("pod")
        if not pod:
            # Pod self-destructed — check if results are on HF
            if _hf_has_checkpoint():
                return {
                    "status": "completed",
                    "pod_id": pod_id,
                    "message": "Pod self-destructed. Checkpoint available on HuggingFace. Run 'terra cloud-download'",
                }
            return {"status": "not_found", "pod_id": pod_id}

        elapsed = time.time() - state.get("started_at", 0)
        runtime = pod.get("runtime") or {}
        gpus = runtime.get("gpus") or []
        gpu_util = gpus[0].get("gpuUtilPercent", "N/A") if gpus else "N/A"
        mem_util = gpus[0].get("memoryUtilPercent", "N/A") if gpus else "N/A"

        status_info = {
            "status": pod.get("desiredStatus", "unknown"),
            "pod_id": pod_id,
            "gpu": state.get("gpu"),
            "uptime_seconds": runtime.get("uptimeInSeconds", 0),
            "elapsed_minutes": round(elapsed / 60, 1),
            "gpu_utilization": gpu_util,
            "memory_utilization": mem_util,
            "max_steps": state.get("max_steps"),
        }

        # Pod self-destructs after training + HF upload
        desired = pod.get("desiredStatus", "")
        if desired in ("EXITED", "STOPPED", "TERMINATED"):
            status_info["status"] = "completed"
            status_info["message"] = "Training done, checkpoint on HuggingFace. Run 'terra cloud-download'"
        elif gpu_util == 0 and runtime.get("uptimeInSeconds", 0) > 120:
            status_info["status"] = "likely_completed"
            status_info["message"] = "Training appears done. Run 'terra cloud-download' to fetch results."

        return status_info

    finally:
        client.close()


def _runpod_download_and_destroy(state: dict, output_dir: str) -> dict:
    """Download results from HuggingFace Hub (pod already self-destructed)."""
    return _download_from_hf(output_dir)


def _runpod_wait_for_completion(client, headers, pod_id, timeout=7200):
    """Wait for training to complete by polling GPU utilization."""
    start = time.time()
    idle_count = 0

    while time.time() - start < timeout:
        result = _runpod_graphql(client, headers, """
            query pod($podId: String!) {
                pod(input: {podId: $podId}) {
                    runtime {
                        uptimeInSeconds
                        gpus { gpuUtilPercent }
                    }
                }
            }
        """, {"podId": pod_id})

        pod = result.get("data", {}).get("pod")
        if not pod or not pod.get("runtime"):
            time.sleep(10)
            continue

        uptime = pod["runtime"].get("uptimeInSeconds", 0)
        gpus = pod["runtime"].get("gpus", [])
        gpu_util = gpus[0].get("gpuUtilPercent", 100) if gpus else 100

        elapsed = round((time.time() - start) / 60, 1)
        print(f"  [{elapsed}m] GPU: {gpu_util}%, uptime: {uptime}s", end="\r")

        # If GPU is idle for several checks, training is probably done
        if gpu_util == 0 and uptime > 120:
            idle_count += 1
            if idle_count >= 3:
                print(f"\n[cloud/runpod] Training complete (GPU idle)")
                return
        else:
            idle_count = 0

        time.sleep(30)

    raise TimeoutError(f"Training did not complete within {timeout}s")


def _runpod_download_results(client, headers, pod_id):
    """Download training results via SSH/rsync."""
    pod_ip = _runpod_get_ssh(client, headers, pod_id)
    if not pod_ip:
        print("[cloud/runpod] WARNING: Could not get SSH access, skipping download")
        return

    ip, port = pod_ip.rsplit(":", 1)
    Path("models/checkpoints/pretrain/final").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync", "-avz", "--progress",
            "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
            f"root@{ip}:/workspace/terra/models/checkpoints/pretrain/final/",
            "models/checkpoints/pretrain/final/",
        ],
        check=True,
    )


def _runpod_get_ssh(client, headers, pod_id) -> str | None:
    """Get SSH connection string for a RunPod pod."""
    result = _runpod_graphql(client, headers, """
        query pod($podId: String!) {
            pod(input: {podId: $podId}) {
                runtime {
                    ports { ip isIpPublic privatePort publicPort }
                }
            }
        }
    """, {"podId": pod_id})

    pod = result.get("data", {}).get("pod")
    if not pod or not pod.get("runtime"):
        return None

    for port in pod["runtime"].get("ports", []):
        if port.get("isIpPublic") and port.get("privatePort") == 22:
            return f"{port['ip']}:{port['publicPort']}"
    return None


def _runpod_destroy(client, headers, pod_id):
    """Terminate a RunPod pod."""
    _runpod_graphql(client, headers,
        "mutation terminatePod($podId: String!) { podTerminate(input: {podId: $podId}) }",
        {"podId": pod_id},
    )


# ---------------------------------------------------------------------------
# Modal (serverless - auto-shutdown, no cleanup needed)
# ---------------------------------------------------------------------------

MODAL_TRAIN_SCRIPT = '''
"""Modal serverless function for Terra pre-training."""
import modal

app = modal.App("terra-pretrain")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "safetensors>=0.4.0",
        "tokenizers>=0.20.0",
        "datasets>=3.0.0",
        "pyyaml>=6.0",
    )
    .copy_local_dir("src", "/app/src")
    .copy_local_dir("configs", "/app/configs")
    .copy_local_dir("models/tokenizer", "/app/models/tokenizer")
    .copy_local_dir("data/pretrain_chunks", "/app/data/pretrain_chunks")
)

volume = modal.Volume.from_name("terra-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="{gpu_type}",
    timeout=7200,
    volumes={{"/checkpoints": volume}},
)
def train(config_yaml: str, max_steps: int, resume_step: int = 0):
    import sys
    sys.path.insert(0, "/app")
    import yaml
    from src.training.pretrain import pretrain

    config = yaml.safe_load(config_yaml)
    arch_config = config.get("architecture", {{}})

    resume_from = None
    if resume_step > 0:
        import os
        ckpt_path = f"/checkpoints/checkpoint-{{resume_step}}/checkpoint.pt"
        if os.path.exists(ckpt_path):
            resume_from = ckpt_path

    result = pretrain(
        model_config=arch_config,
        data_path="/app/data/pretrain_chunks",
        output_dir="/checkpoints",
        batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        max_steps=max_steps,
        warmup_steps=500,
        save_steps=1000,
        eval_steps=500,
        patience=5,
        use_gradient_checkpointing=False,
        resume_from=resume_from,
    )

    volume.commit()
    return result

@app.function(image=image, volumes={{"/checkpoints": volume}})
def download_checkpoint(step: int = 0):
    import os, base64
    path = f"/checkpoints/checkpoint-{{step}}" if step > 0 else "/checkpoints/final"
    files = {{}}
    for f in os.listdir(path):
        fpath = os.path.join(path, f)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as fh:
                files[f] = base64.b64encode(fh.read()).decode()
    return files
'''


def _modal_pretrain(
    config_path: str, gpu: str, max_steps: int, sync_checkpoint: bool, async_mode: bool,
) -> dict:
    """Run pre-training via Modal serverless GPU.

    Modal is inherently async-safe: the function runs on Modal's infrastructure
    independent of your local machine. Even in "sync" mode, if your connection drops
    the job keeps running and results are saved to the persistent volume.
    """
    import yaml

    gpu_map = {
        "A100": "a100-80gb", "H100": "h100", "A10G": "a10g",
        "L4": "l4", "T4": "t4",
    }
    modal_gpu = gpu_map.get(gpu, gpu.lower())

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Write Modal script
    script = MODAL_TRAIN_SCRIPT.replace("{gpu_type}", modal_gpu)
    script_path = Path("src/training/_modal_train.py")
    script_path.write_text(script)

    resume_step = _get_latest_checkpoint_step("models/checkpoints/pretrain")
    config_yaml = yaml.dump(config)

    # Save job state
    _save_cloud_state({
        "provider": "modal",
        "gpu": gpu,
        "max_steps": max_steps,
        "started_at": time.time(),
        "status": "running",
    })

    print(f"[cloud/modal] Starting pre-training on {gpu} ({modal_gpu})...")

    if async_mode:
        # Deploy and detach - Modal keeps running even if we disconnect
        subprocess.Popen(
            [
                "modal", "run", "--detach", str(script_path),
                "--config-yaml", config_yaml,
                "--max-steps", str(max_steps),
                "--resume-step", str(resume_step),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[cloud/modal] Job submitted and detached. Close your laptop safely.")
        print("[cloud/modal] Modal auto-shuts down when done. No pod to destroy.")
        print("[cloud/modal] Check: terra cloud-status")
        print("[cloud/modal] Download: terra cloud-download")
        script_path.unlink(missing_ok=True)
        return {"status": "started", "provider": "modal"}

    # Synchronous mode
    result = subprocess.run(
        [
            "modal", "run", str(script_path),
            "--config-yaml", config_yaml,
            "--max-steps", str(max_steps),
            "--resume-step", str(resume_step),
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )

    script_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Modal training failed: {result.stderr}")

    train_result = json.loads(result.stdout.strip().split("\n")[-1])

    if sync_checkpoint:
        _modal_download({"provider": "modal"}, "models/checkpoints/pretrain/final")
        train_result["model_path"] = "models/checkpoints/pretrain/final"

    CLOUD_STATE_FILE.unlink(missing_ok=True)
    print("[cloud/modal] Pod automatically destroyed (serverless).")
    return train_result


def _modal_check_status(state: dict) -> dict:
    """Check status of a Modal job."""
    elapsed = time.time() - state.get("started_at", 0)
    # Modal doesn't have a simple status API, check if volume has results
    result = subprocess.run(
        ["modal", "volume", "ls", "terra-checkpoints"],
        capture_output=True, text=True,
    )
    has_final = "final/" in result.stdout if result.returncode == 0 else False

    return {
        "status": "completed" if has_final else "running",
        "provider": "modal",
        "gpu": state.get("gpu"),
        "elapsed_minutes": round(elapsed / 60, 1),
        "max_steps": state.get("max_steps"),
        "message": "Results ready! Run 'terra cloud-download'" if has_final else "Still training...",
    }


def _modal_download(state: dict, output_dir: str) -> dict:
    """Download results — try HuggingFace first, fall back to Modal volume."""
    # Try HF first (works for both providers)
    result = _download_from_hf(output_dir)
    if result.get("status") == "downloaded":
        return result

    # Fall back to Modal volume
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dl = subprocess.run(
        ["modal", "volume", "get", "terra-checkpoints", "final/", str(out)],
        capture_output=True, text=True,
    )

    if dl.returncode != 0:
        return {"error": f"Download failed: {dl.stderr}"}

    _copy_to_current(out)
    return {"status": "downloaded", "model_path": str(out)}


# ---------------------------------------------------------------------------
# HuggingFace Hub helpers
# ---------------------------------------------------------------------------

def _hf_has_checkpoint() -> bool:
    """Check if a checkpoint exists on HuggingFace Hub."""
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_REPO_ID")
    if not hf_token or not hf_repo:
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        files = api.list_repo_files(hf_repo)
        return any("latest/" in f for f in files)
    except Exception:
        return False


def _download_from_hf(output_dir: str) -> dict:
    """Download the latest checkpoint from HuggingFace Hub."""
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_REPO_ID")

    if not hf_token or not hf_repo:
        return {"error": "HF_TOKEN and HF_REPO_ID must be set in .env"}

    try:
        from huggingface_hub import snapshot_download

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print(f"[cloud] Downloading from HuggingFace: {hf_repo}/latest ...")
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns="latest/*",
            local_dir=str(out),
            token=hf_token,
        )

        # Move files from latest/ subfolder to output root
        latest_dir = out / "latest"
        if latest_dir.exists():
            import shutil
            for f in latest_dir.iterdir():
                if f.is_file():
                    shutil.move(str(f), out / f.name)
            latest_dir.rmdir()

        _copy_to_current(out)

        return {"status": "downloaded", "model_path": str(out)}

    except Exception as e:
        return {"error": f"HF download failed: {e}"}


def _copy_to_current(source_dir: Path):
    """Copy model files to models/current."""
    import shutil
    current = Path("models/current")
    current.mkdir(parents=True, exist_ok=True)
    for f in source_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, current / f.name)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _get_latest_checkpoint_step(checkpoint_dir: str) -> int:
    """Find the latest checkpoint step in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return 0
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*/checkpoint.pt"))
    if not checkpoints:
        return 0
    return int(checkpoints[-1].parent.name.split("-")[1])


def _fetch_modal_pricing() -> dict[str, float]:
    """Fetch live GPU pricing from Modal's API."""
    import httpx

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get("https://modal.com/pricing")
            text = resp.text

            import re
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', text)
            if match:
                page_data = json.loads(match.group(1))
                props = page_data.get("props", {}).get("pageProps", {})
                gpu_prices = {}
                _extract_modal_gpu_prices(props, gpu_prices)
                if gpu_prices:
                    return gpu_prices
    except Exception:
        pass

    return {}


def _extract_modal_gpu_prices(data: dict | list, result: dict):
    """Recursively search Modal page data for GPU pricing info."""
    if isinstance(data, dict):
        name = data.get("name", "") or data.get("gpu", "") or data.get("label", "")
        price = data.get("price", None) or data.get("pricePerHour", None) or data.get("cost", None)
        if name and price and isinstance(price, (int, float)):
            normalized = _normalize_gpu_name(str(name))
            if normalized:
                result[normalized] = float(price)
        for v in data.values():
            if isinstance(v, (dict, list)):
                _extract_modal_gpu_prices(v, result)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _extract_modal_gpu_prices(item, result)


def _fetch_runpod_pricing() -> dict[str, float]:
    """Fetch live GPU pricing from RunPod's API."""
    import httpx

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        return {}

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                "https://api.runpod.io/graphql",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": """
                    query GpuTypes {
                        gpuTypes {
                            id
                            displayName
                            memoryInGb
                            communityPrice
                            securePrice
                        }
                    }
                    """,
                },
            )
            resp.raise_for_status()
            gpu_types = resp.json().get("data", {}).get("gpuTypes", [])

            prices = {}
            for gpu in gpu_types:
                name = _normalize_gpu_name(gpu.get("displayName", ""))
                price = gpu.get("securePrice") or gpu.get("communityPrice")
                if name and price:
                    prices[name] = round(float(price), 3)
            return prices
    except Exception:
        return {}


def _normalize_gpu_name(raw: str) -> str | None:
    """Normalize GPU name to our standard keys."""
    raw_lower = raw.lower().replace("-", "").replace(" ", "")
    mapping = {
        "a10080gb": "A100", "a100": "A100", "a10080gbpcie": "A100", "a10080gbsxm": "A100",
        "nvidiaa10080gb": "A100",
        "h100": "H100", "h10080gb": "H100", "h10080gbhbm3": "H100", "nvidiah10080gb": "H100",
        "a10g": "A10G", "nvidiaa10g": "A10G",
        "l4": "L4", "nvidial4": "L4",
        "t4": "T4", "nvidiat4": "T4",
        "rtx5090": "RTX5090", "nvidiagefortxrtx5090": "RTX5090",
        "rtx4090": "RTX4090", "nvidiagefortxrtx4090": "RTX4090",
        "rtx3090": "RTX3090", "rtxa6000": "A6000", "a6000": "A6000",
    }
    for pattern, normalized in mapping.items():
        if pattern in raw_lower:
            return normalized
    return None


# Hardcoded fallbacks (only used when APIs are unreachable)
# Speed = seconds per training step (measured from actual runs, NOT tokens/sec)
_FALLBACK_STEP_TIME = {
    "A100": 1.2, "H100": 0.8, "H200": 0.7, "A10G": 2.5,
    "L4": 4.0, "T4": 8.0, "RTX5090": 0.7, "RTX4090": 1.5,
}
_FALLBACK_PRICING = {
    "A100": 3.40, "H100": 4.76, "H200": 3.39, "A10G": 1.10,
    "L4": 0.80, "T4": 0.53, "RTX5090": 0.77, "RTX4090": 0.69,
}
# Overhead for data download, tokenizer training, pip install, model upload (minutes)
_OVERHEAD_MINUTES = {
    "pretrain": 10,
    "sft": 10,
    "multimodal": 30,  # includes image + audio tokenizer training
}


def estimate_cost(
    gpu: str,
    max_steps: int,
    mode: str = "pretrain",
    provider: str | None = None,
    gpu_count: int = 1,
) -> dict:
    """Estimate cloud training cost based on measured step times.

    Uses seconds-per-step from actual training runs + fixed overhead
    for data download, tokenizer training, pip install, and model upload.
    Multi-GPU scales step time down linearly (data parallelism) but costs more per hour.
    """
    live_prices = {}
    pricing_source = "fallback"

    if provider in (None, "runpod"):
        rp_prices = _fetch_runpod_pricing()
        if rp_prices:
            live_prices.update(rp_prices)
            pricing_source = "runpod_api"

    if provider in (None, "modal"):
        modal_prices = _fetch_modal_pricing()
        if modal_prices:
            live_prices.update(modal_prices)
            pricing_source = "modal_api"

    cost_per_hr = live_prices.get(gpu, _FALLBACK_PRICING.get(gpu, 2.0))
    if gpu not in live_prices:
        pricing_source = "fallback"

    secs_per_step = _FALLBACK_STEP_TIME.get(gpu, 1.0)
    # Multi-GPU: each step processes gpu_count x more data, so fewer steps needed
    # But we keep step count the same — each step is faster due to data parallelism
    # In practice with DDP: step time stays ~same but effective batch is larger
    # So for same number of steps, time is similar but throughput is gpu_count x higher
    # We model this as: training time stays ~same, cost = time x gpu_count x price
    overhead_min = _OVERHEAD_MINUTES.get(mode, 10)
    training_hours = (max_steps * secs_per_step) / 3600
    total_hours = training_hours + (overhead_min / 60)
    cost = total_hours * cost_per_hr * gpu_count

    return {
        "gpu": gpu,
        "gpu_count": gpu_count,
        "estimated_hours": round(total_hours, 2),
        "estimated_cost_usd": round(cost, 2),
        "cost_per_hour": round(cost_per_hr * gpu_count, 3),
        "training_hours": round(training_hours, 2),
        "overhead_minutes": overhead_min,
        "secs_per_step": secs_per_step,
        "pricing_source": pricing_source,
    }
