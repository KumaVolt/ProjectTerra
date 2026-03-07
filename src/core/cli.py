"""CLI entrypoint for ProjectTerra."""

import json
from pathlib import Path

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
    num_samples: int = typer.Option(1000000, help="Number of text samples to train on"),
    output_dir: str = typer.Option("models/tokenizer", help="Output directory"),
    diverse: bool = typer.Option(True, help="Use diverse data mix (English + code + multilingual)"),
):
    """Train a BPE tokenizer from scratch on diverse data."""
    from src.training.tokenizer import train_from_datasets

    console.print(f"[bold]Training tokenizer[/bold] (vocab_size={vocab_size}, samples={num_samples}, diverse={diverse})")
    tokenizer = train_from_datasets(
        vocab_size=vocab_size,
        num_samples=num_samples,
        output_dir=output_dir,
        diverse=diverse,
    )
    console.print(f"[bold green]Tokenizer trained: {tokenizer.get_vocab_size()} tokens[/bold green]")

    # Auto-benchmark
    from src.training.tokenizer import benchmark_tokenizer
    results = benchmark_tokenizer(output_dir)
    console.print(f"\n[bold]Tokenizer Quality:[/bold]")
    for domain, stats in results.items():
        tpw = stats['tokens_per_word']
        color = "green" if tpw < 1.5 else "yellow" if tpw < 2.0 else "red"
        console.print(f"  {domain:15s} [{color}]{tpw:.2f} tok/word[/{color}] ({stats['tokens']} tokens for {stats['words']} words)")
        if "digit_tokens" in stats:
            console.print(f"                  digits: {stats['digit_tokens']}")


@app.command()
def benchmark_tok(
    tokenizer_path: str = typer.Option("models/tokenizer", help="Tokenizer path"),
):
    """Benchmark tokenizer quality on different domains."""
    from src.training.tokenizer import benchmark_tokenizer

    results = benchmark_tokenizer(tokenizer_path)
    console.print(f"[bold]Tokenizer Benchmark[/bold] ({tokenizer_path})\n")
    for domain, stats in results.items():
        tpw = stats['tokens_per_word']
        color = "green" if tpw < 1.5 else "yellow" if tpw < 2.0 else "red"
        console.print(f"  {domain:15s} [{color}]{tpw:.2f} tok/word[/{color}]  |  {stats['tokens']} tokens / {stats['words']} words / {stats['chars']} chars")
        if "digit_tokens" in stats:
            console.print(f"                  digit splitting: {stats['digit_tokens']}")


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

    est = estimate_cost(gpu, display_steps, mode="pretrain", provider=provider)
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
def cloud_sft(
    gpu: str = typer.Option("A100", help="GPU type: A100, H100, L4, RTX4090"),
    max_steps: int = typer.Option(0, help="Training steps (0 = auto, 3 epochs)"),
    max_samples: int = typer.Option(20000, help="Max SFT training samples"),
    estimate_only: bool = typer.Option(False, "--estimate", help="Only show cost estimate"),
    async_mode: bool = typer.Option(True, "--async/--sync", help="Async mode (safe to close laptop)"),
):
    """Fine-tune on cloud GPU. Downloads base model from HF, runs SFT, uploads result."""
    from src.training.cloud import estimate_cost

    # SFT is faster than pre-training — estimate ~2000 steps
    display_steps = max_steps if max_steps > 0 else 2000

    est = estimate_cost(gpu, display_steps, mode="sft")
    console.print(f"[bold]Cloud SFT estimate:[/bold]")
    console.print(f"  GPU: {est['gpu']}")
    console.print(f"  Rate: ${est['cost_per_hour']}/hr")
    console.print(f"  Time: ~{est['estimated_hours']} hours")
    console.print(f"  Cost: ~${est['estimated_cost_usd']:.2f}")
    console.print(f"  SFT samples: {max_samples}")
    if async_mode:
        console.print(f"  Mode: [cyan]async (safe to close laptop)[/cyan]")

    if estimate_only:
        return

    if not typer.confirm(f"\nProceed with ~${est['estimated_cost_usd']:.2f} cloud SFT?"):
        console.print("Cancelled.")
        return

    # Launch cloud pod with SFT mode
    from src.training.cloud import _runpod_launch_pod

    result = _runpod_launch_pod(
        gpu=gpu,
        max_steps=max_steps,
        async_mode=async_mode,
        mode="sft",
        extra_env=[
            {"key": "SFT_MAX_SAMPLES", "value": str(max_samples)},
        ],
    )

    if async_mode:
        console.print(f"\n[bold green]SFT job submitted![/bold green]")
        console.print("  The pod will:")
        console.print("  1. Download base model from HuggingFace")
        console.print("  2. Download SlimOrca instruction data")
        console.print("  3. Fine-tune and upload to HF as 'sft-latest'")
        console.print("  4. Self-destruct")
        console.print("\n  Check progress: terra cloud-status")
        console.print("  Download results: terra cloud-download")


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
def chat(
    model_path: str = typer.Argument(None, help="Path to model checkpoint"),
    max_tokens: int = typer.Option(200, help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    tokenizer_path: str = typer.Option("models/tokenizer", help="Path to tokenizer"),
):
    """Interactive chat with your Terra model."""
    import torch
    from pathlib import Path

    from src.training.model import TerraForCausalLM
    from src.training.tokenizer import load_tokenizer

    # Auto-detect model path
    if model_path is None:
        candidates = [
            "models/current",
            "models/checkpoints/pretrain/best/latest",
            "models/checkpoints/pretrain/best",
            "models/checkpoints/pretrain/final",
        ]
        for c in candidates:
            if Path(c).exists() and (Path(c) / "config.json").exists():
                model_path = c
                break
        if model_path is None:
            console.print("[red]No model found. Train one first with 'terra pretrain' or 'terra cloud-train'.[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Loading Terra from {model_path}...[/bold]")
    model = TerraForCausalLM.from_pretrained(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Model loaded ({params / 1e6:.1f}M params). Type your prompt, empty line to quit.[/green]")
    console.print("[dim]Tip: This is a base model (text completion). For Q&A, try prompts like:[/dim]")
    console.print("[dim]  'Question: What is 5+5? Answer:' or 'The capital of France is'[/dim]\n")

    while True:
        try:
            prompt = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            break

        input_ids = torch.tensor([tokenizer.encode(prompt).ids])
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        # Only decode the NEW tokens (skip the prompt)
        new_tokens = output[0][prompt_len:].tolist()
        response = tokenizer.decode(new_tokens).strip()
        console.print(f"[bold green]Terra:[/bold green] {response}\n")

    console.print("\n[dim]Goodbye![/dim]")


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
def sft(
    model_path: str = typer.Argument(None, help="Path to pre-trained model (auto-detects)"),
    data_dir: str = typer.Option("data/sft", help="SFT data directory"),
    output_dir: str = typer.Option("models/checkpoints/sft", help="Output directory"),
    batch_size: int = typer.Option(4, help="Micro batch size"),
    grad_accum: int = typer.Option(4, help="Gradient accumulation steps"),
    max_steps: int = typer.Option(0, help="Max steps (0 = auto, 3 epochs)"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
    max_length: int = typer.Option(1024, help="Max sequence length"),
    download: bool = typer.Option(True, help="Download SFT data if not present"),
    max_samples: int = typer.Option(20000, help="Max training samples to download"),
):
    """Instruction fine-tune Terra (SFT). Makes the model conversational."""
    from src.training.sft import download_sft_data, finetune

    # Auto-detect model
    if model_path is None:
        candidates = [
            "models/checkpoints/pretrain/best/latest",
            "models/checkpoints/pretrain/best",
            "models/checkpoints/pretrain/final",
            "models/current",
        ]
        for c in candidates:
            if Path(c).exists() and (Path(c) / "config.json").exists():
                model_path = c
                break
        if model_path is None:
            console.print("[red]No pre-trained model found. Run 'terra pretrain' or 'terra cloud-train' first.[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Instruction fine-tuning (SFT)[/bold]")
    console.print(f"  Base model: {model_path}")

    # Download data if needed
    data_file = Path(data_dir) / "train.jsonl"
    if download and not data_file.exists():
        console.print(f"[bold cyan]Downloading instruction data...[/bold cyan]")
        download_sft_data(output_dir=data_dir, max_samples=max_samples)

    if not data_file.exists():
        console.print("[red]No SFT data found. Download failed or was skipped.[/red]")
        raise typer.Exit(1)

    result = finetune(
        model_path=model_path,
        data_path=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=max_steps,
        max_length=max_length,
    )

    console.print(f"\n[bold green]SFT complete![/bold green]")
    console.print(f"  Best val loss: {result.get('best_val_loss', 'N/A'):.4f}")
    console.print(f"  Model saved to: {result.get('model_path', output_dir)}")
    console.print(f"  Try it: terra chat")


@app.command()
def train_vision(
    data_dir: str = typer.Option("data/vision", help="Vision data directory"),
    output_dir: str = typer.Option("models/checkpoints/vision", help="Output directory"),
    preset: str = typer.Option("vision_tiny", help="Preset: vision_tiny, vision_small, vision_base"),
    batch_size: int = typer.Option(32, help="Batch size"),
    max_steps: int = typer.Option(5000, help="Max training steps (0 = auto)"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
):
    """Train the vision encoder (image understanding)."""
    from src.training.train_multimodal import train_vision_encoder

    console.print(f"[bold]Training vision encoder ({preset})...[/bold]")
    result = train_vision_encoder(
        data_dir=data_dir, output_dir=output_dir, preset=preset,
        batch_size=batch_size, learning_rate=learning_rate, max_steps=max_steps,
    )
    console.print(f"[bold green]Vision training complete.[/bold green]")
    console.print(f"  Steps: {result['total_steps']}, Best val loss: {result.get('best_val_loss', 'N/A')}")


@app.command()
def train_image_gen(
    data_dir: str = typer.Option("data/image_gen", help="Image gen data directory"),
    output_dir: str = typer.Option("models/checkpoints/image_gen", help="Output directory"),
    preset: str = typer.Option("gen_tiny", help="Preset: gen_tiny, gen_small, gen_base"),
    batch_size: int = typer.Option(8, help="Batch size"),
    max_steps: int = typer.Option(10000, help="Total steps (VAE + diffusion)"),
    vae_steps: int = typer.Option(2000, help="Steps for VAE pre-training phase"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
):
    """Train the image generator (text-to-image diffusion)."""
    from src.training.train_multimodal import train_image_generator

    console.print(f"[bold]Training image generator ({preset})...[/bold]")
    result = train_image_generator(
        data_dir=data_dir, output_dir=output_dir, preset=preset,
        batch_size=batch_size, learning_rate=learning_rate,
        max_steps=max_steps, vae_pretrain_steps=vae_steps,
    )
    console.print(f"[bold green]Image generator training complete.[/bold green]")
    console.print(f"  Steps: {result['total_steps']}")


@app.command()
def train_audio(
    data_dir: str = typer.Option("data/audio", help="Audio data directory"),
    output_dir: str = typer.Option("models/checkpoints/audio", help="Output directory"),
    preset: str = typer.Option("audio_tiny", help="Preset: audio_tiny, audio_small, audio_base"),
    batch_size: int = typer.Option(8, help="Batch size"),
    max_steps: int = typer.Option(5000, help="Max training steps (0 = auto)"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
):
    """Train the audio encoder (speech-to-text)."""
    from src.training.train_multimodal import train_audio_encoder

    console.print(f"[bold]Training audio encoder ({preset})...[/bold]")
    result = train_audio_encoder(
        data_dir=data_dir, output_dir=output_dir, preset=preset,
        batch_size=batch_size, learning_rate=learning_rate, max_steps=max_steps,
    )
    console.print(f"[bold green]Audio encoder training complete.[/bold green]")
    console.print(f"  Steps: {result['total_steps']}, Best val loss: {result.get('best_val_loss', 'N/A')}")


@app.command()
def train_speech(
    data_dir: str = typer.Option("data/tts", help="TTS data directory"),
    output_dir: str = typer.Option("models/checkpoints/speech", help="Output directory"),
    preset: str = typer.Option("speech_tiny", help="Preset: speech_tiny, speech_small"),
    batch_size: int = typer.Option(8, help="Batch size"),
    max_steps: int = typer.Option(5000, help="Max training steps (0 = auto)"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
):
    """Train the speech decoder (text-to-speech codec)."""
    from src.training.train_multimodal import train_speech_decoder

    console.print(f"[bold]Training speech decoder ({preset})...[/bold]")
    result = train_speech_decoder(
        data_dir=data_dir, output_dir=output_dir, preset=preset,
        batch_size=batch_size, learning_rate=learning_rate, max_steps=max_steps,
    )
    console.print(f"[bold green]Speech decoder training complete.[/bold green]")
    console.print(f"  Steps: {result['total_steps']}, Best val loss: {result.get('best_val_loss', 'N/A')}")


@app.command()
def test_vision(
    image_path: str = typer.Argument(..., help="Path to an image file"),
    model_path: str = typer.Option("models/checkpoints/vision/best", help="Vision encoder checkpoint"),
):
    """Test vision encoder: describe what the model sees in an image."""
    import torch
    from PIL import Image
    from torchvision import transforms

    from src.training.vision_encoder import TerraVisionEncoder

    console.print(f"[bold]Loading vision encoder from {model_path}...[/bold]")
    model = TerraVisionEncoder.from_pretrained(model_path)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((model.config.image_size, model.config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0)

    with torch.no_grad():
        embeddings = model(pixel_values)

    console.print(f"[green]Image encoded into {embeddings.shape[1]} tokens of dimension {embeddings.shape[2]}[/green]")
    console.print(f"Embedding norm: {embeddings.norm(dim=-1).mean().item():.4f}")
    console.print("(These embeddings would be fed into the Terra LLM for captioning/VQA)")


@app.command()
def test_image_gen(
    prompt: str = typer.Argument(..., help="Text prompt for image generation"),
    output_path: str = typer.Option("generated.png", help="Output image path"),
    model_path: str = typer.Option("models/checkpoints/image_gen", help="Image generator checkpoint"),
    steps: int = typer.Option(4, help="Denoising steps (1-50, fewer = faster)"),
):
    """Generate an image from a text prompt."""
    import torch

    from src.training.image_generator import ImageGenConfig, TerraImageGenerator
    from src.training.tokenizer import load_tokenizer

    console.print(f"[bold]Loading image generator from {model_path}...[/bold]")

    config = ImageGenConfig.from_dict(json.loads(Path(model_path, "config.json").read_text()))
    model = TerraImageGenerator(config)
    state = torch.load(str(Path(model_path) / "image_generator.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tokenizer = load_tokenizer("models/tokenizer")
    token_ids = tokenizer.encode(prompt).ids[:77]
    token_ids = token_ids + [0] * (77 - len(token_ids))
    input_ids = torch.tensor([token_ids])

    # Simple text embedding (would normally come from the LLM)
    text_embed = torch.nn.Embedding(32000, config.context_dim)
    context = text_embed(input_ids)

    console.print(f"Generating with {steps} steps...")
    with torch.no_grad():
        images = model.generate_fast(context, num_steps=steps) if steps <= 4 else model.generate(context, num_steps=steps)

    # Save as PNG
    from torchvision.utils import save_image
    save_image(images[0] * 0.5 + 0.5, output_path)  # [-1,1] -> [0,1]
    console.print(f"[green]Image saved to {output_path}[/green]")


@app.command()
def test_stt(
    audio_path: str = typer.Argument(..., help="Path to a WAV audio file"),
    model_path: str = typer.Option("models/checkpoints/audio/best", help="Audio encoder checkpoint"),
):
    """Transcribe speech from an audio file (speech-to-text)."""
    import torch
    import torchaudio

    from src.data.multimodal_downloader import _compute_mel
    from src.training.audio_encoder import TerraAudioEncoder
    from src.training.tokenizer import load_tokenizer

    console.print(f"[bold]Loading audio encoder from {model_path}...[/bold]")
    model = TerraAudioEncoder.from_pretrained(model_path)
    model.eval()

    tokenizer = load_tokenizer("models/tokenizer")

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform[0]  # mono

    # Compute mel
    mel = _compute_mel(waveform, 16000).unsqueeze(0)

    with torch.no_grad():
        log_probs = model.forward_ctc(mel)  # (1, T, vocab)
        pred_ids = log_probs.argmax(dim=-1)[0].tolist()

    # CTC decode: collapse repeats, remove blanks
    decoded = []
    prev = -1
    for idx in pred_ids:
        if idx != 0 and idx != prev:
            decoded.append(idx)
        prev = idx

    text = tokenizer.decode(decoded)
    console.print(f"[green]Transcription:[/green] {text}")


@app.command()
def test_tts(
    text: str = typer.Argument(..., help="Text to synthesize"),
    output_path: str = typer.Option("speech.wav", help="Output WAV file path"),
    model_path: str = typer.Option("models/checkpoints/speech/best", help="Speech decoder checkpoint"),
):
    """Synthesize speech from text (text-to-speech)."""
    import struct
    import wave

    import torch

    from src.data.multimodal_downloader import _compute_mel
    from src.training.speech_decoder import TerraSpeechDecoder
    from src.training.tokenizer import load_tokenizer

    console.print(f"[bold]Loading speech decoder from {model_path}...[/bold]")
    model = TerraSpeechDecoder.from_pretrained(model_path)
    model.eval()

    tokenizer = load_tokenizer("models/tokenizer")

    # Encode text -> simulate LLM hidden states
    token_ids = tokenizer.encode(text).ids
    input_tensor = torch.tensor([token_ids])
    text_embed = torch.nn.Embedding(32000, model.config.lm_hidden_size)
    lm_hidden = text_embed(input_tensor)

    # Generate: encode a reference mel to get tokens, then decode conditioned on text
    # For now, just demonstrate the codec pipeline with a dummy mel
    target_frames = len(token_ids) * 10  # rough estimate
    dummy_mel = torch.randn(1, model.config.num_mel_bins, target_frames)

    with torch.no_grad():
        # Codec encode -> VQ -> decode (shows the reconstruction quality)
        mel_hat, token_ids_vq, _ = model.forward_codec(dummy_mel)

    # Convert mel to simple waveform (Griffin-Lim approximation)
    mel_np = mel_hat[0].exp().numpy()
    # Simple overlap-add synthesis
    sr = model.config.sample_rate
    hop = model.config.hop_length
    n_frames = mel_np.shape[1]
    audio = torch.zeros(n_frames * hop)
    for i in range(n_frames):
        freq = mel_np[:, i].mean() * 0.01
        t = torch.linspace(0, hop / sr, hop)
        audio[i * hop:(i + 1) * hop] = torch.sin(2 * 3.14159 * 440 * freq * t) * 0.3

    # Save WAV
    audio_int16 = (audio.clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

    console.print(f"[green]Speech saved to {output_path}[/green]")
    console.print("[dim](Note: quality depends on training. Untrained model produces noise.)[/dim]")


@app.command()
def train_image_tok(
    data_dir: str = typer.Option("data/vision", help="Vision data directory"),
    output_dir: str = typer.Option("models/checkpoints/image_tokenizer", help="Output directory"),
    batch_size: int = typer.Option(16, help="Batch size"),
    max_steps: int = typer.Option(10000, help="Max training steps (0 = auto)"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
    image_size: int = typer.Option(256, help="Image size"),
    codebook_size: int = typer.Option(8192, help="VQ codebook size"),
):
    """Train the image VQ-VAE tokenizer (unified architecture)."""
    from src.training.train_multimodal import train_image_tokenizer

    console.print(f"[bold]Training image VQ-VAE tokenizer (codebook={codebook_size})...[/bold]")
    result = train_image_tokenizer(
        data_dir=data_dir, output_dir=output_dir,
        batch_size=batch_size, learning_rate=learning_rate,
        max_steps=max_steps, image_size=image_size,
        codebook_size=codebook_size,
    )
    console.print(f"[bold green]Image tokenizer training complete.[/bold green]")
    console.print(f"  Steps: {result['total_steps']}, Best val loss: {result.get('best_val_loss', 'N/A')}")


@app.command()
def train_audio_tok(
    data_dir: str = typer.Option("data/tts", help="Audio data directory"),
    output_dir: str = typer.Option("models/checkpoints/audio_tokenizer", help="Output directory"),
    batch_size: int = typer.Option(8, help="Batch size"),
    max_steps: int = typer.Option(5000, help="Max training steps (0 = auto)"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
):
    """Train the audio codec tokenizer (unified architecture)."""
    from src.training.train_multimodal import train_audio_tokenizer

    console.print(f"[bold]Training audio codec tokenizer...[/bold]")
    result = train_audio_tokenizer(
        data_dir=data_dir, output_dir=output_dir,
        batch_size=batch_size, learning_rate=learning_rate,
        max_steps=max_steps,
    )
    console.print(f"[bold green]Audio tokenizer training complete.[/bold green]")
    console.print(f"  Steps: {result['total_steps']}, Best val loss: {result.get('best_val_loss', 'N/A')}")


@app.command()
def train_multimodal(
    text_model_path: str = typer.Option(None, help="Pre-trained text model path (auto-detects)"),
    image_tokenizer_path: str = typer.Option("models/checkpoints/image_tokenizer/best", help="Image tokenizer path"),
    audio_tokenizer_path: str = typer.Option("models/checkpoints/audio_tokenizer/best", help="Audio tokenizer path"),
    vision_dir: str = typer.Option("data/vision", help="Vision data directory"),
    audio_dir: str = typer.Option("data/tts", help="Audio data directory"),
    output_dir: str = typer.Option("models/checkpoints/multimodal", help="Output directory"),
    batch_size: int = typer.Option(4, help="Micro batch size"),
    grad_accum: int = typer.Option(4, help="Gradient accumulation steps"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    max_steps: int = typer.Option(10000, help="Max training steps (0 = auto)"),
    max_seq_len: int = typer.Option(512, help="Max sequence length"),
    depth_loss_weight: float = typer.Option(1.0, help="Weight for depth transformer loss"),
):
    """Multimodal fine-tuning: train unified model on interleaved text+image+audio."""
    from src.training.train_multimodal import train_multimodal as run_mm

    # Auto-detect text model
    if text_model_path is None:
        candidates = [
            "models/current",
            "models/checkpoints/sft/best",
            "models/checkpoints/pretrain/best/latest",
            "models/checkpoints/pretrain/best",
            "models/checkpoints/pretrain/final",
        ]
        for c in candidates:
            if Path(c).exists() and (Path(c) / "config.json").exists():
                text_model_path = c
                break
        if text_model_path is None:
            console.print("[red]No text model found. Run 'terra pretrain' or 'terra sft' first.[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Multimodal fine-tuning[/bold]")
    console.print(f"  Text model: {text_model_path}")
    console.print(f"  Image tokenizer: {image_tokenizer_path}")
    console.print(f"  Audio tokenizer: {audio_tokenizer_path}")

    result = run_mm(
        text_model_path=text_model_path,
        image_tokenizer_path=image_tokenizer_path,
        audio_tokenizer_path=audio_tokenizer_path,
        vision_dir=vision_dir,
        audio_dir=audio_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_steps=max_steps,
        max_seq_len=max_seq_len,
        depth_loss_weight=depth_loss_weight,
    )

    console.print(f"\n[bold green]Multimodal fine-tuning complete![/bold green]")
    console.print(f"  Steps: {result.get('total_steps', '?')}")
    console.print(f"  Best val loss: {result.get('best_val_loss', 'N/A')}")
    console.print(f"  Model saved to: {result.get('model_path', output_dir)}")


@app.command()
def cloud_multimodal(
    gpu: str = typer.Option("A100", help="GPU type: A100, H100, L4, RTX4090"),
    max_steps: int = typer.Option(0, help="Training steps (0 = auto, 10K)"),
    estimate_only: bool = typer.Option(False, "--estimate", help="Only show cost estimate"),
    async_mode: bool = typer.Option(True, "--async/--sync", help="Async mode (safe to close laptop)"),
):
    """Train multimodal model on cloud GPU. Trains tokenizers + unified model."""
    from src.training.cloud import estimate_cost

    display_steps = max_steps if max_steps > 0 else 10000

    est = estimate_cost(gpu, display_steps, mode="multimodal")
    console.print(f"[bold]Cloud multimodal training estimate:[/bold]")
    console.print(f"  GPU: {est['gpu']}")
    console.print(f"  Rate: ${est['cost_per_hour']}/hr")
    console.print(f"  Time: ~{est['estimated_hours']} hours")
    console.print(f"  Cost: ~${est['estimated_cost_usd']:.2f}")
    console.print(f"  Pipeline: image_tok -> audio_tok -> multimodal finetune")
    if async_mode:
        console.print(f"  Mode: [cyan]async (safe to close laptop)[/cyan]")

    if estimate_only:
        return

    if not typer.confirm(f"\nProceed with ~${est['estimated_cost_usd']:.2f} cloud multimodal training?"):
        console.print("Cancelled.")
        return

    from src.training.cloud import _runpod_launch_pod

    result = _runpod_launch_pod(
        gpu=gpu,
        max_steps=max_steps,
        async_mode=async_mode,
        mode="multimodal",
        extra_env=[],
    )

    if async_mode:
        console.print(f"\n[bold green]Multimodal job submitted![/bold green]")
        console.print("  The pod will:")
        console.print("  1. Download base model + multimodal data")
        console.print("  2. Train image & audio tokenizers")
        console.print("  3. Run multimodal fine-tuning")
        console.print("  4. Upload to HF and self-destruct")
        console.print("\n  Check progress: terra cloud-status")
        console.print("  Download results: terra cloud-download")


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
