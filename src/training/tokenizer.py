"""Train a BPE tokenizer from scratch for Terra.

Uses HuggingFace tokenizers library for fast training.
Produces a sentencepiece-compatible tokenizer with special tokens
for chat, code, and multimodal markers.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


# Special tokens for Terra
SPECIAL_TOKENS = [
    "<|pad|>",        # 0 - padding
    "<|bos|>",        # 1 - beginning of sequence
    "<|eos|>",        # 2 - end of sequence
    "<|unk|>",        # 3 - unknown
    "<|im_start|>",   # 4 - chat turn start
    "<|im_end|>",     # 5 - chat turn end
    "<|system|>",     # 6 - system message
    "<|user|>",       # 7 - user message
    "<|assistant|>",  # 8 - assistant message
    "<|code|>",       # 9 - code block start
    "<|/code|>",      # 10 - code block end
    "<|image|>",      # 11 - image placeholder (for vision)
    "<|audio|>",      # 12 - audio placeholder (for speech)
    "<|think|>",      # 13 - reasoning/thinking start
    "<|/think|>",     # 14 - reasoning/thinking end
]


def create_tokenizer(vocab_size: int = 32000) -> Tokenizer:
    """Create an untrained BPE tokenizer with Terra's configuration."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Normalizer: NFC unicode normalization
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenizer: split on whitespace + digits + punctuation (GPT-4 style)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r"'s|'t|'re|'ve|'m|'ll|'d", behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train_tokenizer(
    data_paths: list[str],
    vocab_size: int = 32000,
    output_dir: str = "models/tokenizer",
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a BPE tokenizer on the given text files.

    Args:
        data_paths: List of paths to text files or directories for training.
        vocab_size: Target vocabulary size.
        output_dir: Where to save the trained tokenizer.
        min_frequency: Minimum frequency for a token to be included.

    Returns:
        Trained Tokenizer instance.
    """
    tokenizer = create_tokenizer(vocab_size)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Collect all text files
    files = []
    for p in data_paths:
        path = Path(p)
        if path.is_file():
            files.append(str(path))
        elif path.is_dir():
            files.extend(str(f) for f in path.glob("**/*.txt"))
            files.extend(str(f) for f in path.glob("**/*.jsonl"))

    if not files:
        raise ValueError(f"No training files found in: {data_paths}")

    print(f"Training tokenizer on {len(files)} files, target vocab_size={vocab_size}")
    tokenizer.train(files, trainer)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))

    # Save config for easy loading
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: i for i, tok in enumerate(SPECIAL_TOKENS)},
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2))

    print(f"Tokenizer saved to {out} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer."""
    tokenizer_path = Path(path) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"No tokenizer found at {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def train_from_datasets(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    vocab_size: int = 32000,
    num_samples: int = 100000,
    output_dir: str = "models/tokenizer",
) -> Tokenizer:
    """Train tokenizer directly from a HuggingFace dataset (streaming)."""
    from datasets import load_dataset

    print(f"Loading {dataset_name}/{subset} for tokenizer training ({num_samples} samples)...")
    ds = load_dataset(dataset_name, subset, split="train", streaming=True)

    tokenizer = create_tokenizer(vocab_size)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def batch_iterator(dataset, batch_size=1000):
        batch = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            text = example.get("text", "")
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    tokenizer.train_from_iterator(batch_iterator(ds), trainer, length=num_samples)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))

    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: i for i, tok in enumerate(SPECIAL_TOKENS)},
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2))

    print(f"Tokenizer saved to {out} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer
