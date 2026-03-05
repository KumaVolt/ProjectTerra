"""CLI entrypoint for ProjectTerra."""

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console

app = typer.Typer(name="terra", help="ProjectTerra - Self-evolving multimodal LLM (from scratch)")
console = Console()


@app.command()
def evolve(
    config: str = typer.Option("configs/terra.yaml", help="Path to config file"),
):
    """Run one evolution session."""
    from src.core.orchestrator import EvolutionOrchestrator

    console.print("[bold green]ProjectTerra Evolution Session[/bold green]")
    orchestrator = EvolutionOrchestrator(config)
    orchestrator.run()
    console.print("[bold green]Session complete.[/bold green]")


@app.command()
def train_tokenizer(
    vocab_size: int = typer.Option(32000, help="Vocabulary size"),
    num_samples: int = typer.Option(100000, help="Number of text samples to train on"),
    output_dir: str = typer.Option("models/tokenizer", help="Output directory"),
):
    """Train a BPE tokenizer from scratch."""
    from src.training.tokenizer import train_from_datasets

    console.print(f"[bold]Training tokenizer[/bold] (vocab_size={vocab_size}, samples={num_samples})")
    tokenizer = train_from_datasets(
        vocab_size=vocab_size,
        num_samples=num_samples,
        output_dir=output_dir,
    )
    console.print(f"[bold green]Tokenizer trained: {tokenizer.get_vocab_size()} tokens[/bold green]")


@app.command()
def download_data(
    output_dir: str = typer.Option("data/pretrain", help="Output directory"),
    max_samples: int = typer.Option(50000, help="Max samples per source"),
    minimal: bool = typer.Option(False, help="Download minimal sample for testing"),
):
    """Download pre-training datasets."""
    from src.data.downloader import download_minimal_sample, download_pretraining_data

    if minimal:
        console.print("[bold]Downloading minimal sample for local testing...[/bold]")
        result = download_minimal_sample(output_dir)
    else:
        console.print("[bold]Downloading pre-training data...[/bold]")
        result = download_pretraining_data(output_dir, max_samples)

    for name, path in result.items():
        console.print(f"  {name}: {path}")
    console.print("[bold green]Download complete.[/bold green]")


@app.command()
def prepare_data(
    data_dir: str = typer.Option("data/pretrain", help="Raw data directory"),
    output_dir: str = typer.Option("data/pretrain_chunks", help="Output directory"),
    chunk_size: int = typer.Option(2048, help="Sequence length per chunk"),
    tokenizer_path: str = typer.Option("models/tokenizer", help="Tokenizer path"),
):
    """Tokenize and chunk pre-training data."""
    from src.data.downloader import prepare_pretraining_chunks

    console.print("[bold]Preparing pre-training chunks...[/bold]")
    result = prepare_pretraining_chunks(data_dir, output_dir, chunk_size, tokenizer_path)
    console.print(f"[bold green]Data prepared: {result}[/bold green]")


@app.command()
def pretrain(
    config: str = typer.Option("configs/terra.yaml", help="Path to config file"),
    max_steps: int = typer.Option(None, help="Override max training steps (0 = auto based on data size)"),
    resume: str = typer.Option(None, help="Resume from checkpoint path"),
):
    """Pre-train Terra model from scratch."""
    import yaml
    from src.training.pretrain import pretrain as run_pretrain

    with open(config) as f:
        cfg = yaml.safe_load(f)

    arch_config = cfg.get("architecture", {})
    pretrain_config = cfg["training"].get("pretrain", {})

    if max_steps is not None:
        pretrain_config["max_steps_per_session"] = max_steps

    params_est = None
    from src.training.model import TerraConfig
    tc = TerraConfig.from_dict(arch_config)
    params_est = tc.param_count_estimate()

    console.print(f"[bold]Pre-training Terra ({params_est / 1e6:.0f}M params estimated)[/bold]")

    result = run_pretrain(
        model_config=arch_config,
        data_path=pretrain_config.get("data_path", "data/pretrain_chunks"),
        output_dir="models/checkpoints/pretrain",
        batch_size=pretrain_config.get("batch_size", 4),
        gradient_accumulation_steps=pretrain_config.get("gradient_accumulation_steps", 8),
        learning_rate=pretrain_config.get("learning_rate", 3e-4),
        max_steps=pretrain_config.get("max_steps_per_session", 2000),
        warmup_steps=pretrain_config.get("warmup_steps", 200),
        save_steps=pretrain_config.get("save_steps", 500),
        use_gradient_checkpointing=pretrain_config.get("gradient_checkpointing", True),
        resume_from=resume,
    )

    console.print(f"\n[bold green]Pre-training complete.[/bold green]")
    console.print(f"  Steps: {result['total_steps']}")
    console.print(f"  Final loss: {result.get('final_loss', 'N/A')}")
    console.print(f"  Model saved to: {result['model_path']}")


@app.command()
def cloud_train(
    config: str = typer.Option("configs/terra.yaml", help="Path to config file"),
    provider: str = typer.Option(None, help="Cloud provider: 'modal' or 'runpod' (auto-detects)"),
    gpu: str = typer.Option("A100", help="GPU type: A100, H100, A10G, L4, T4, RTX4090"),
    max_steps: int = typer.Option(0, help="Training steps (0 = auto based on data size)"),
    estimate_only: bool = typer.Option(False, "--estimate", help="Only show cost estimate"),
    async_mode: bool = typer.Option(False, "--async", help="Fire-and-forget (safe to close laptop)"),
):
    """Pre-train on a cloud GPU. Creates pod, trains, downloads results, destroys pod."""
    from src.training.cloud import cloud_pretrain, estimate_cost

    # For auto mode, calculate steps from data to show meaningful estimate
    display_steps = max_steps
    if max_steps <= 0:
        from pathlib import Path
        data_path = Path("data/pretrain_chunks/train.jsonl")
        if data_path.exists():
            num_chunks = sum(1 for _ in open(data_path))
            cloud_batch = 32 * 2  # batch_size * grad_accum on cloud
            steps_per_epoch = max(1, num_chunks // cloud_batch)
            display_steps = steps_per_epoch * 3  # 3 epochs
            console.print(f"[bold]Auto steps:[/bold] {num_chunks} chunks -> {steps_per_epoch} steps/epoch -> {display_steps} steps (3 epochs)")
        else:
            display_steps = 1000  # rough fallback
            console.print("[yellow]No local data found to calculate steps. Using estimate of ~1000 steps.[/yellow]")

    est = estimate_cost(gpu, display_steps, provider=provider)
    source = est['pricing_source']
    source_label = {
        "runpod_api": "[green]live from RunPod API[/green]",
        "modal_api": "[green]live from Modal API[/green]",
        "fallback": "[yellow]estimated (APIs unreachable)[/yellow]",
    }.get(source, source)

    console.print(f"[bold]Cloud training estimate:[/bold]")
    console.print(f"  GPU: {est['gpu']}")
    console.print(f"  Rate: ${est['cost_per_hour']}/hr ({source_label})")
    console.print(f"  Time: ~{est['estimated_hours']} hours")
    console.print(f"  Cost: ~${est['estimated_cost_usd']:.2f}")
    console.print(f"  Tokens: {est['total_tokens']:,}")
    if max_steps <= 0:
        console.print(f"  Steps: [cyan]auto ({display_steps}, with early stopping)[/cyan]")
    if async_mode:
        console.print(f"  Mode: [cyan]async (safe to close laptop)[/cyan]")

    if estimate_only:
        return

    if not typer.confirm(f"\nProceed with ~${est['estimated_cost_usd']:.2f} cloud training?"):
        console.print("Cancelled.")
        return

    result = cloud_pretrain(config, provider=provider, gpu=gpu, max_steps=max_steps, async_mode=async_mode)

    if async_mode:
        console.print(f"\n[bold green]Job submitted![/bold green]")
        console.print("  Check progress:   terra cloud-status")
        console.print("  Download results: terra cloud-download")
    else:
        console.print(f"\n[bold green]Cloud training complete.[/bold green]")
        if "model_path" in result:
            console.print(f"  Model saved to: {result['model_path']}")


@app.command()
def cloud_status():
    """Check the status of an active cloud training job."""
    from src.training.cloud import cloud_status as check_status

    result = check_status()

    if result.get("status") == "no_active_job":
        console.print("No active cloud training job.")
        return

    console.print(f"[bold]Cloud Job Status[/bold]")
    console.print(f"  Provider: {result.get('provider', 'unknown')}")
    console.print(f"  GPU: {result.get('gpu', 'unknown')}")
    console.print(f"  Status: {result.get('status', 'unknown')}")
    console.print(f"  Elapsed: {result.get('elapsed_minutes', '?')} minutes")

    if "gpu_utilization" in result:
        console.print(f"  GPU util: {result['gpu_utilization']}%")
        console.print(f"  Memory util: {result.get('memory_utilization', '?')}%")

    if result.get("max_steps"):
        console.print(f"  Target steps: {result['max_steps']}")

    if result.get("message"):
        console.print(f"\n  {result['message']}")


@app.command()
def cloud_download(
    output_dir: str = typer.Option("models/checkpoints/pretrain/final", help="Where to save"),
):
    """Download results from a completed cloud job and destroy the pod."""
    from src.training.cloud import cloud_download as download

    console.print("[bold]Downloading cloud training results...[/bold]")
    result = download(output_dir)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    console.print(f"[bold green]Download complete.[/bold green]")
    console.print(f"  Model saved to: {result.get('model_path', output_dir)}")
    if result.get("pod_destroyed"):
        console.print("  Pod destroyed (no longer billing).")


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to model to evaluate"),
    config: str = typer.Option("configs/terra.yaml", help="Path to config file"),
    quick: bool = typer.Option(True, help="Run quick eval with limited samples"),
):
    """Run benchmarks on a model."""
    import yaml
    from src.evaluation.benchmarks import BenchmarkRunner

    with open(config) as f:
        cfg = yaml.safe_load(f)

    runner = BenchmarkRunner(cfg)
    if quick:
        results = runner.run_quick_eval(model_path)
    else:
        results = runner.run_full_eval(model_path)

    console.print_json(data=results)
    score = runner.compute_weighted_score(results)
    console.print(f"\n[bold]Weighted Score: {score:.4f}[/bold]")


@app.command()
def generate_data(
    domain: str = typer.Argument(..., help="Domain to generate data for"),
    samples: int = typer.Option(50, help="Number of samples"),
    config: str = typer.Option("configs/terra.yaml", help="Path to config file"),
):
    """Generate training data using available LLMs (Claude, GLM, or both)."""
    import yaml
    from src.core.llm_client import LLMPool
    from src.data.generator import DataGenerator

    with open(config) as f:
        cfg = yaml.safe_load(f)

    pool = LLMPool()
    console.print(f"Available LLMs: {pool.available()}")

    gen = DataGenerator(cfg, pool)
    data = gen.generate_distillation_data(domain, samples)
    path = gen.save_data(data, f"manual_{domain}")
    console.print(f"Generated {len(data)} samples, saved to {path}")


@app.command()
def serve(
    model_path: str = typer.Argument("models/current", help="Path to model"),
    port: int = typer.Option(8080, help="Server port"),
):
    """Start a local inference server for the terra model (for self-hosting)."""
    import subprocess

    console.print(f"Starting terra model server on port {port}...")
    console.print(f"This enables TERRA_SERVER_URL=http://localhost:{port}")
    try:
        subprocess.run(
            ["python", "-m", "mlx_lm.server", "--model", model_path, "--port", str(port)],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        console.print("MLX not available, trying llama.cpp server...")
        subprocess.run(
            ["llama-server", "-m", model_path, "--port", str(port)],
            check=True,
        )


@app.command()
def status():
    """Show current evolution status."""
    import json
    from pathlib import Path

    state_path = Path("configs/evolution_state.json")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        console.print("[bold]Evolution State[/bold]")
        console.print_json(data=state)
    else:
        console.print("No evolution state found. Run 'terra evolve' to start.")

    # Check for tokenizer
    tok_path = Path("models/tokenizer/tokenizer.json")
    console.print(f"\nTokenizer: {'[green]ready[/green]' if tok_path.exists() else '[red]not trained[/red]'}")

    # Check for pre-training data
    data_path = Path("data/pretrain_chunks/train.jsonl")
    console.print(f"Pre-training data: {'[green]ready[/green]' if data_path.exists() else '[red]not prepared[/red]'}")

    # Check for model
    model_path = Path("models/current/config.json")
    console.print(f"Current model: {'[green]ready[/green]' if model_path.exists() else '[yellow]not yet trained[/yellow]'}")

    # Show architecture info
    from src.training.model import TerraConfig
    config = TerraConfig.terra_150m()
    est = config.param_count_estimate()
    console.print(f"\nTarget architecture: terra-150m (~{est / 1e6:.0f}M params)")

    from src.core.session_logger import SessionLogger
    logger = SessionLogger()
    recent = logger.get_recent_sessions(5)
    if recent:
        console.print(f"\n[bold]Recent Sessions ({len(recent)}):[/bold]")
        for s in recent:
            console.print(
                f"  {s['session_id']} - {s['status']} - "
                f"stages: {', '.join(s.get('stages_completed', []))}"
            )


@app.command()
def init():
    """Initialize the full pipeline: train tokenizer, download data, prepare chunks."""
    console.print("[bold]Initializing Terra from-scratch pipeline...[/bold]\n")

    # Step 1: Train tokenizer
    console.print("[bold cyan]Step 1/3: Training tokenizer...[/bold cyan]")
    from src.training.tokenizer import train_from_datasets
    tokenizer = train_from_datasets(num_samples=50000)
    console.print(f"  Tokenizer ready: {tokenizer.get_vocab_size()} tokens\n")

    # Step 2: Download data
    console.print("[bold cyan]Step 2/3: Downloading pre-training data...[/bold cyan]")
    from src.data.downloader import download_minimal_sample
    result = download_minimal_sample()
    for name, path in result.items():
        console.print(f"  {name}: {path}")
    console.print()

    # Step 3: Prepare chunks
    console.print("[bold cyan]Step 3/3: Tokenizing and chunking data...[/bold cyan]")
    from src.data.downloader import prepare_pretraining_chunks
    prepare_pretraining_chunks()
    console.print()

    console.print("[bold green]Pipeline initialized! Run 'terra pretrain' to start training.[/bold green]")


@app.command()
def download_multimodal(
    modality: str = typer.Argument("all", help="Which modality: all, vision, image-gen, audio, tts"),
    minimal: bool = typer.Option(True, help="Download minimal sample for testing"),
    max_samples: int = typer.Option(None, help="Override max samples"),
):
    """Download training data for multimodal components."""
    from src.data.multimodal_downloader import (
        download_all_multimodal_data,
        download_audio_data,
        download_image_gen_data,
        download_tts_data,
        download_vision_data,
    )

    if modality == "all":
        console.print("[bold]Downloading all multimodal training data...[/bold]")
        results = download_all_multimodal_data(minimal=minimal)
        for name, r in results.items():
            console.print(f"  {name}: {r.get('total', '?')} samples")
    elif modality == "vision":
        console.print("[bold]Downloading vision data...[/bold]")
        download_vision_data(max_samples=max_samples or (5000 if minimal else 50000), minimal=minimal)
    elif modality == "image-gen":
        console.print("[bold]Downloading image generation data...[/bold]")
        download_image_gen_data(max_samples=max_samples or (1000 if minimal else 50000), minimal=minimal)
    elif modality == "audio":
        console.print("[bold]Downloading audio/STT data...[/bold]")
        download_audio_data(max_samples=max_samples or (2000 if minimal else 20000), minimal=minimal)
    elif modality == "tts":
        console.print("[bold]Downloading TTS data...[/bold]")
        download_tts_data(max_samples=max_samples or (2000 if minimal else 20000), minimal=minimal)
    else:
        console.print(f"[red]Unknown modality: {modality}. Use: all, vision, image-gen, audio, tts[/red]")

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
