"""Pre-training pipeline for Terra model from scratch.

Supports:
- Local training on MacBook Air M4 (MPS backend)
- Cloud GPU bursts via Modal/RunPod
- Gradient checkpointing for memory efficiency
- Mixed precision (bfloat16)
- Cosine LR schedule with warmup
- Early stopping on validation loss (prevents overfitting on limited data)
- Logging and checkpointing
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class PretrainDataset(Dataset):
    """Dataset of pre-tokenized chunks for pre-training."""

    def __init__(self, data_path: str):
        self.chunks = []
        path = Path(data_path)

        if path.is_dir():
            files = list(path.glob("*.jsonl"))
        else:
            files = [path]

        for f in files:
            with open(f) as fh:
                for line in fh:
                    data = json.loads(line)
                    self.chunks.append(torch.tensor(data["input_ids"], dtype=torch.long))

        print(f"Loaded {len(self.chunks)} training chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine annealing with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_val_loss(model, val_dataloader, device, dtype) -> float:
    """Compute average loss on validation set."""
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            output = model(input_ids=input_ids, labels=labels)
        total_loss += output["loss"].item()
        count += 1
        if count >= 50:  # Cap eval at 50 batches to keep it fast
            break
    model.train()
    return total_loss / max(count, 1)


def pretrain(
    model_config: dict,
    data_path: str,
    output_dir: str = "models/checkpoints/pretrain",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    max_steps: int = 10000,
    warmup_steps: int = 500,
    save_steps: int = 1000,
    eval_steps: int = 200,
    log_steps: int = 10,
    max_grad_norm: float = 1.0,
    use_gradient_checkpointing: bool = True,
    resume_from: str | None = None,
    val_split: float = 0.05,
    patience: int = 5,
) -> dict:
    """Run pre-training from scratch.

    Args:
        model_config: TerraConfig as dict, or a preset name like "terra_150m".
        data_path: Path to chunked pre-training data.
        output_dir: Where to save checkpoints.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        max_steps: Total training steps.
        warmup_steps: LR warmup steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate validation loss every N steps.
        log_steps: Log metrics every N steps.
        max_grad_norm: Gradient clipping.
        use_gradient_checkpointing: Trade compute for memory.
        resume_from: Path to checkpoint to resume from.
        val_split: Fraction of data to hold out for validation.
        patience: Stop after N evals with no improvement (0 = disabled).

    Returns:
        Training results dict.
    """
    from src.training.model import TerraConfig, TerraForCausalLM

    device = get_device()
    print(f"Device: {device}")

    # Build model
    if isinstance(model_config, str):
        config_fn = getattr(TerraConfig, model_config, None)
        if config_fn:
            config = config_fn()
        else:
            raise ValueError(f"Unknown model preset: {model_config}")
    else:
        config = TerraConfig.from_dict(model_config)

    model = TerraForCausalLM(config)
    params = model.count_parameters()
    print(f"Model: {params['total_millions']:.1f}M parameters")

    # Gradient checkpointing for memory savings
    if use_gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    model = model.to(device)

    # Use bfloat16 on supported devices
    dtype = torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float32
    if device.type == "mps":
        dtype = torch.float32  # MPS bfloat16 support is limited

    # Dataset with train/val split
    full_dataset = PretrainDataset(data_path)
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Track epochs and auto-calculate max_steps if needed
    steps_per_epoch = max(1, train_size // (batch_size * gradient_accumulation_steps))

    if max_steps <= 0:
        # Auto mode: 3 epochs over the data (with early stopping as safety net)
        max_epochs = 3
        max_steps = steps_per_epoch * max_epochs
        warmup_steps = min(warmup_steps, max_steps // 10)
        print(f"Auto max_steps: {max_steps} ({max_epochs} epochs x {steps_per_epoch} steps/epoch)")

    total_epochs = max_steps / steps_per_epoch
    print(f"Data: {train_size} train chunks, {val_size} val chunks")
    print(f"Steps per epoch: {steps_per_epoch}, total epochs: {total_epochs:.1f}")
    if total_epochs > 4:
        print(f"WARNING: {total_epochs:.1f} epochs over the same data. Consider adding more data or reducing max_steps.")
        # Auto-cap at 4 epochs to prevent overfitting
        max_steps = min(max_steps, steps_per_epoch * 4)
        print(f"Auto-capped max_steps to {max_steps} (4 epochs)")

    # Scale eval/save intervals to data size (eval ~2x per epoch, save ~1x per epoch)
    if eval_steps <= 0 or eval_steps > steps_per_epoch:
        eval_steps = max(10, steps_per_epoch // 2)
    if save_steps <= 0 or save_steps > max_steps:
        save_steps = max(50, steps_per_epoch)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "norm" in name or "bias" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler = get_cosine_schedule(optimizer, warmup_steps, max_steps)

    # Resume from checkpoint
    start_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step} (best val_loss: {best_val_loss:.4f})")

    # Training loop
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.train()
    step = start_step
    epoch = 0
    total_loss = 0.0
    tokens_processed = 0
    start_time = time.time()
    log_history = []
    stop_reason = "max_steps"

    data_iter = iter(train_dataloader)

    print(f"Starting pre-training: {max_steps} steps, batch={batch_size}, accum={gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    if patience > 0:
        print(f"Early stopping: patience={patience} evals, eval every {eval_steps} steps")

    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                print(f"--- Epoch {epoch} complete ---")
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                output = model(input_ids=input_ids, labels=labels)
                loss = output["loss"] / gradient_accumulation_steps

            loss.backward()
            accum_loss += loss.item()
            tokens_processed += input_ids.numel()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        step += 1
        total_loss += accum_loss

        # Logging
        if step % log_steps == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            avg_loss = total_loss / log_steps
            lr = scheduler.get_last_lr()[0]

            log_entry = {
                "step": step,
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "lr": round(lr, 8),
                "grad_norm": round(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, 4),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "tokens_total": tokens_processed,
            }
            log_history.append(log_entry)
            print(
                f"step {step}/{max_steps} | epoch {epoch} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                f"grad_norm {log_entry['grad_norm']:.2f} | {tokens_per_sec:.0f} tok/s"
            )
            total_loss = 0.0

        # Validation + early stopping
        if step % eval_steps == 0:
            val_loss = evaluate_val_loss(model, val_dataloader, device, dtype)
            print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

            log_history.append({"step": step, "val_loss": round(val_loss, 4)})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                _save_checkpoint(model, optimizer, scheduler, step, config, out, log_history, best_val_loss)
                model.save_pretrained(str(out / "best"))
                print(f"  New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")

                if patience > 0 and patience_counter >= patience:
                    print(f"Early stopping: val_loss hasn't improved for {patience} evals")
                    stop_reason = "early_stopping"
                    break

        # Periodic checkpointing
        elif step % save_steps == 0:
            _save_checkpoint(model, optimizer, scheduler, step, config, out, log_history, best_val_loss)

    # Final save
    _save_checkpoint(model, optimizer, scheduler, step, config, out, log_history, best_val_loss)
    model.save_pretrained(str(out / "final"))

    # Copy best model as the "final" if early stopping triggered
    best_dir = out / "best"
    if stop_reason == "early_stopping" and best_dir.exists():
        import shutil
        final_dir = out / "final"
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.copytree(best_dir, final_dir)
        print("Using best checkpoint as final model (early stopping)")

    elapsed = time.time() - start_time
    result = {
        "total_steps": step,
        "epochs_completed": epoch,
        "final_loss": log_history[-1].get("loss", log_history[-2].get("loss")) if len(log_history) >= 2 else None,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "stop_reason": stop_reason,
        "total_tokens": tokens_processed,
        "training_time_seconds": round(elapsed, 1),
        "tokens_per_second": round(tokens_processed / elapsed, 1) if elapsed > 0 else 0,
        "model_path": str(out / "final"),
        "parameter_count": params["total"],
    }

    (out / "training_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nPre-training complete: {result}")
    return result


def _save_checkpoint(model, optimizer, scheduler, step, config, out_dir, log_history, best_val_loss=None):
    """Save a training checkpoint."""
    ckpt_path = out_dir / f"checkpoint-{step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.to_dict(),
    }
    if best_val_loss is not None:
        save_dict["best_val_loss"] = best_val_loss

    torch.save(save_dict, str(ckpt_path / "checkpoint.pt"))

    # Save log history
    (ckpt_path / "log_history.json").write_text(json.dumps(log_history, indent=2))
    print(f"  Checkpoint saved: {ckpt_path}")


def _enable_gradient_checkpointing(model):
    """Enable gradient checkpointing on transformer layers."""
    for i, layer in enumerate(model.model.layers):
        def make_ckpt_forward(orig_forward):
            def ckpt_forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)
            return ckpt_forward

        layer.forward = make_ckpt_forward(layer.forward)
