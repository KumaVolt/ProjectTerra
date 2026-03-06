"""Supervised Fine-Tuning (SFT) for Terra.

Takes a pre-trained base model and fine-tunes it on instruction-following data.
This teaches the model to understand and respond to questions, follow instructions,
and have conversations instead of just completing text.

Data format: conversations with system/user/assistant turns.
Loss is only computed on assistant responses (the model learns to ANSWER, not parrot).
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Special tokens used in conversation format
CHAT_TEMPLATE = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end_turn": "<|end|>",
}


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning on instruction data."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        path = Path(data_path)
        if path.is_file():
            self._load_file(path)
        elif path.is_dir():
            for f in sorted(path.glob("*.jsonl")):
                self._load_file(f)

        print(f"[sft] Loaded {len(self.examples)} training examples")

    def _load_file(self, path: Path):
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    self.examples.append(entry)
                except json.JSONDecodeError:
                    continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        entry = self.examples[idx]
        conversations = entry.get("conversations", [])

        # Build the full text with role markers
        text_parts = []
        # Track where assistant responses start/end for loss masking
        assistant_ranges = []

        full_text = ""
        for turn in conversations:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))

            # Normalize role names
            if role in ("system",):
                marker = CHAT_TEMPLATE["system"]
            elif role in ("user", "human"):
                marker = CHAT_TEMPLATE["user"]
            elif role in ("assistant", "gpt"):
                marker = CHAT_TEMPLATE["assistant"]
            else:
                continue

            turn_text = f"{marker}\n{content}{CHAT_TEMPLATE['end_turn']}\n"

            if role in ("assistant", "gpt"):
                start = len(full_text) + len(f"{marker}\n")
                end = start + len(content) + len(CHAT_TEMPLATE["end_turn"])
                assistant_ranges.append((start, end))

            full_text += turn_text

        # Tokenize the full conversation
        encoded = self.tokenizer.encode(full_text)
        input_ids = encoded.ids[:self.max_length]

        # Create labels: -100 for non-assistant tokens (ignored in loss)
        labels = [-100] * len(input_ids)

        # Find which token positions correspond to assistant responses
        # by tokenizing up to each character position
        if assistant_ranges:
            # Tokenize prefix to find token boundaries
            for a_start, a_end in assistant_ranges:
                # Get token index for start of assistant response
                prefix_before = full_text[:a_start]
                tok_start = len(self.tokenizer.encode(prefix_before).ids)

                prefix_after = full_text[:a_end]
                tok_end = len(self.tokenizer.encode(prefix_after).ids)

                # Set labels for assistant tokens
                for i in range(tok_start, min(tok_end, len(input_ids))):
                    labels[i] = input_ids[i]

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def download_sft_data(
    output_dir: str = "data/sft",
    max_samples: int = 20000,
) -> str:
    """Download instruction-following data for SFT.

    Uses SlimOrca — high-quality GPT-4 generated instruction data.
    """
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / "train.jsonl"

    if output_file.exists():
        count = sum(1 for _ in open(output_file))
        print(f"[sft] Data already exists: {count} examples at {output_file}")
        return str(output_file)

    print(f"[sft] Downloading SlimOrca instruction data (up to {max_samples} samples)...")
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)

    count = 0
    with open(output_file, "w") as f:
        for example in ds:
            if count >= max_samples:
                break

            conversations = example.get("conversations", [])
            if not conversations:
                continue

            # Normalize to standard format
            normalized = []
            for turn in conversations:
                role = turn.get("from", "")
                value = turn.get("value", "")
                if role == "system":
                    normalized.append({"role": "system", "content": value})
                elif role == "human":
                    normalized.append({"role": "user", "content": value})
                elif role == "gpt":
                    normalized.append({"role": "assistant", "content": value})

            if normalized and any(t["role"] == "assistant" for t in normalized):
                f.write(json.dumps({"conversations": normalized}) + "\n")
                count += 1

                if count % 5000 == 0:
                    print(f"  [sft] {count}/{max_samples}")

    print(f"[sft] Saved {count} examples to {output_file}")
    return str(output_file)


def finetune(
    model_path: str,
    data_path: str = "data/sft",
    output_dir: str = "models/checkpoints/sft",
    tokenizer_path: str = "models/tokenizer",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_steps: int = 0,
    num_epochs: int = 3,
    max_length: int = 1024,
    warmup_ratio: float = 0.05,
    eval_steps: int = 0,
    save_steps: int = 0,
    patience: int = 3,
) -> dict:
    """Fine-tune a pre-trained Terra model on instruction data.

    Args:
        model_path: Path to pre-trained model checkpoint.
        data_path: Path to SFT data (JSONL with conversations).
        output_dir: Where to save the fine-tuned model.
        tokenizer_path: Path to tokenizer.
        batch_size: Micro batch size.
        gradient_accumulation_steps: Effective batch = batch_size * this.
        learning_rate: Lower than pre-training (2e-5 vs 3e-4).
        max_steps: 0 = auto (num_epochs * steps_per_epoch).
        num_epochs: Number of passes over the data.
        max_length: Max sequence length for training examples.
        warmup_ratio: Fraction of steps for LR warmup.
        eval_steps: Evaluate every N steps (0 = once per epoch).
        save_steps: Save every N steps (0 = same as eval_steps).
        patience: Early stopping patience (number of evals without improvement).
    """
    from src.training.model import TerraForCausalLM
    from src.training.tokenizer import load_tokenizer

    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[sft] Device: {device}, dtype: {dtype}")

    # Load model
    print(f"[sft] Loading model from {model_path}...")
    model = TerraForCausalLM.from_pretrained(model_path).to(device)
    params = model.count_parameters()
    print(f"[sft] Model: {params['total_millions']:.1f}M params")

    # Load tokenizer and data
    tokenizer = load_tokenizer(tokenizer_path)
    dataset = SFTDataset(data_path, tokenizer, max_length=max_length)

    if len(dataset) == 0:
        raise RuntimeError(f"No training data found at {data_path}. Run download first.")

    # Train/val split
    val_size = max(1, min(500, int(len(dataset) * 0.05)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Calculate steps
    steps_per_epoch = max(1, len(train_loader) // gradient_accumulation_steps)
    if max_steps <= 0:
        max_steps = steps_per_epoch * num_epochs
    if eval_steps <= 0:
        eval_steps = steps_per_epoch
    if save_steps <= 0:
        save_steps = eval_steps

    warmup_steps = int(max_steps * warmup_ratio)

    print(f"[sft] Data: {len(train_ds)} train, {len(val_ds)} val")
    print(f"[sft] Effective batch: {batch_size * gradient_accumulation_steps}")
    print(f"[sft] Steps: {max_steps} ({num_epochs} epochs x {steps_per_epoch} steps/epoch)")
    print(f"[sft] Warmup: {warmup_steps} steps, LR: {learning_rate}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Training loop
    step = 0
    best_val_loss = float("inf")
    evals_without_improvement = 0
    start_time = time.time()
    epoch = 0
    accum_loss = 0.0
    accum_steps = 0

    model.train()
    print(f"\n[sft] Starting fine-tuning...")

    while step < max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            output = model(input_ids)
            logits = output["logits"]

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Loss only on assistant tokens (labels != -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accum_loss += loss.item()
            accum_steps += 1

            if accum_steps >= gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                avg_loss = accum_loss
                accum_loss = 0.0
                accum_steps = 0

                if step % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    print(f"  step {step}/{max_steps} | epoch {epoch} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s")

                # Eval
                if step % eval_steps == 0:
                    val_loss = _eval_sft(model, val_loader, device)
                    print(f"  val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        evals_without_improvement = 0
                        model.save_pretrained(str(out / "best"))
                        print(f"  New best model saved!")
                    else:
                        evals_without_improvement += 1
                        if evals_without_improvement >= patience:
                            print(f"  Early stopping: no improvement for {patience} evals")
                            step = max_steps  # exit
                            break

                    model.train()

        print(f"--- Epoch {epoch} complete ---")

    # Save final
    elapsed = time.time() - start_time
    model.save_pretrained(str(out / "final"))

    # Copy best to models/current for easy access
    best_path = out / "best"
    current_path = Path("models/current")
    if best_path.exists():
        import shutil
        if current_path.exists():
            shutil.rmtree(current_path)
        shutil.copytree(best_path, current_path)
        print(f"[sft] Best model copied to models/current/")

    result = {
        "total_steps": step,
        "epochs_completed": epoch,
        "best_val_loss": best_val_loss,
        "training_time_seconds": elapsed,
        "parameter_count": params["total"],
        "model_path": str(out / "best"),
    }

    print(f"\n[sft] Fine-tuning complete!")
    print(f"  Steps: {step}, Best val loss: {best_val_loss:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Model saved to: {out / 'best'}")
    print(f"  Also copied to: models/current/")

    return result


def _eval_sft(model, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids)
            logits = output["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)
