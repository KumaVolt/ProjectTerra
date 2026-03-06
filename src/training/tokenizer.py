"""Train a BPE tokenizer from scratch for Terra.

Uses HuggingFace tokenizers library for fast training.
Key improvements over naive BPE:
1. GPT-4 style pre-tokenization with explicit digit splitting
2. Diverse training data: English + code + multilingual
3. Large training corpus (1M+ samples) for better merge statistics
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


# Special tokens for Terra
#
# Design philosophy (aligned with Qwen3):
# - Special tokens ONLY for things that must be atomic single tokens:
#   control flow, chat structure, and modality boundaries.
# - Everything else (tool calling, thinking, code blocks, JSON) is plain text
#   using Qwen3/Hermes-compatible tags like <tool_call>, <think>, etc.
#   This lets us directly use existing training data without conversion.
#
# The model learns plain-text tags like <tool_call> as regular token sequences,
# same as Qwen3 does. This is realistic for a 150M model — we need every
# public dataset we can get, and they all use text-based formats.
#
# Order matters: index = token ID.
SPECIAL_TOKENS = [
    # Core control (0-3)
    "<|pad|>",           # 0 - padding
    "<|bos|>",           # 1 - beginning of sequence
    "<|eos|>",           # 2 - end of sequence
    "<|unk|>",           # 3 - unknown

    # Chat structure (4-5) — Qwen3 compatible
    "<|im_start|>",      # 4 - chat turn start (role follows as text)
    "<|im_end|>",        # 5 - chat turn end / EOS for generation

    # Multimodal boundaries (6-15) — these MUST be special tokens because
    # they mark where the token space switches between text and discrete
    # codebook tokens. The model needs an unambiguous single-token signal
    # to know "the next N tokens are image/audio/video codes, not text".
    "<|image|>",         # 6  - image token sequence start
    "<|/image|>",        # 7  - image token sequence end
    "<|audio|>",         # 8  - audio token sequence start
    "<|/audio|>",        # 9  - audio token sequence end
    "<|user_audio|>",    # 10 - full-duplex: user audio stream marker
    "<|agent_audio|>",   # 11 - full-duplex: agent audio stream marker
    "<|video|>",         # 12 - video token sequence start
    "<|/video|>",        # 13 - video token sequence end
    "<|3d|>",            # 14 - 3D/point cloud token sequence start
    "<|/3d|>",           # 15 - 3D/point cloud token sequence end

    # Reserved (16-23) — room for future modalities or features
    *[f"<|reserved_{i}|>" for i in range(8)],
]

# Plain-text tags used during training/inference (NOT special tokens).
# These are regular text that gets tokenized into subwords, same as Qwen3.
# Listed here as constants for consistency across the codebase.
TEXT_TAGS = {
    # Chat roles (used after <|im_start|>)
    "system": "system",
    "user": "user",
    "assistant": "assistant",

    # Reasoning (Qwen3 compatible)
    "think_start": "<think>",
    "think_end": "</think>",

    # Tool calling (Hermes/Qwen3 compatible — can use public training data directly)
    "tool_call_start": "<tool_call>",
    "tool_call_end": "</tool_call>",
    "tool_response_start": "<tool_response>",
    "tool_response_end": "</tool_response>",
    "tools_start": "<tools>",
    "tools_end": "</tools>",

    # Code blocks
    "code_start": "```",
    "code_end": "```",
}


def create_tokenizer(vocab_size: int = 32000) -> Tokenizer:
    """Create an untrained BPE tokenizer with Terra's configuration.

    Key design choices:
    - Individual digit splitting via Digits(individual_digits=True)
      Forces each digit to be its own pre-token so BPE can't merge "123" into
      one token. This is crucial for arithmetic and reasoning.
    - Contraction handling ('s, 't, 're, etc.)
    - ByteLevel fallback (no unknown tokens ever)
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Normalizer: NFC unicode normalization (canonical form for accents etc.)
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenizer pipeline:
    # 1. Split contractions so they become separate tokens
    # 2. Split individual digits so "12345" -> "1" "2" "3" "4" "5"
    # 3. ByteLevel ensures full byte coverage (no UNK tokens)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r"'s|'t|'re|'ve|'m|'ll|'d", behavior="isolated"),
        pre_tokenizers.Digits(individual_digits=True),
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


def _build_diverse_iterator(num_samples: int, batch_size: int = 500):
    """Build an iterator over diverse data for tokenizer training.

    Collects data to a temp file first, then iterates — avoids memory issues
    from holding multiple streaming dataset connections open simultaneously.
    """
    import gc
    import tempfile

    from datasets import load_dataset

    sources = [
        ("english",        "HuggingFaceFW/fineweb-edu", "sample-10BT", "text",    0.40),
        ("code",           "codeparrot/codeparrot-clean", None,         "content", 0.20),
        ("chinese",        "wikimedia/wikipedia",    "20231101.zh",     "text",    0.10),
        ("spanish",        "wikimedia/wikipedia",    "20231101.es",     "text",    0.05),
        ("tool_use",       "glaiveai/glaive-function-calling-v2", None, "chat",    0.10),
        ("reasoning",      "openai/gsm8k",           "main",           "answer",  0.05),
    ]
    # Remaining 10% filled by fallback to english

    # Phase 1: collect all text to a temp file (one line per sample)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="terra_tok_")
    total_collected = 0

    for name, dataset, subset, field, frac in sources:
        target = int(num_samples * frac)
        print(f"  [{name}] loading {target} samples from {dataset}...", end=" ", flush=True)
        try:
            kwargs = {"split": "train", "streaming": True}
            ds = load_dataset(dataset, subset, **kwargs) if subset else load_dataset(dataset, **kwargs)

            count = 0
            for example in ds:
                if count >= target:
                    break
                text = example.get(field, "")
                if text and len(text) > 20:
                    # Cap length, replace newlines to keep one-line-per-sample
                    tmp.write(text[:5000].replace("\n", " ") + "\n")
                    count += 1
            print(f"{count} samples")
            total_collected += count

            # Free the dataset iterator to avoid memory buildup
            del ds
            gc.collect()

        except Exception as e:
            print(f"FAILED: {e}")
            # Fallback: extra English data
            if name != "english":
                print(f"    -> falling back to English for {target} samples...", end=" ", flush=True)
                try:
                    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                                     split="train", streaming=True)
                    count = 0
                    for example in ds:
                        if count >= target:
                            break
                        text = example.get("text", "")
                        if text and len(text) > 20:
                            tmp.write(text[:5000].replace("\n", " ") + "\n")
                            count += 1
                    print(f"{count} samples")
                    total_collected += count
                    del ds
                    gc.collect()
                except Exception:
                    print("also failed")

    tmp.close()
    print(f"  Total collected: {total_collected} samples -> {tmp.name}")

    # Phase 2: iterate the temp file in batches
    with open(tmp.name) as f:
        batch = []
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # Cleanup
    import os
    os.unlink(tmp.name)


def train_from_datasets(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    vocab_size: int = 32000,
    num_samples: int = 100000,
    output_dir: str = "models/tokenizer",
    diverse: bool = True,
) -> Tokenizer:
    """Train tokenizer on diverse data sources.

    Args:
        diverse: If True, train on English + code + multilingual mix.
                 If False, train only on the specified dataset (legacy behavior).
    """
    tokenizer = create_tokenizer(vocab_size)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    if diverse and num_samples >= 10000:
        print(f"Training tokenizer on diverse data mix ({num_samples} total samples)...")
        print(f"  Mix: 50% English, 30% code, 20% multilingual")
        iterator = _build_diverse_iterator(num_samples)
    else:
        from datasets import load_dataset
        print(f"Loading {dataset_name}/{subset} for tokenizer training ({num_samples} samples)...")
        ds = load_dataset(dataset_name, subset, split="train", streaming=True)

        def iterator():
            batch = []
            for i, example in enumerate(ds):
                if i >= num_samples:
                    break
                text = example.get("text", "")
                if text:
                    batch.append(text)
                if len(batch) >= 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch

        iterator = iterator()

    tokenizer.train_from_iterator(iterator, trainer, length=num_samples)

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


def benchmark_tokenizer(tokenizer_path: str = "models/tokenizer") -> dict:
    """Benchmark tokenizer quality on standard test strings.

    Returns compression stats: tokens per word, tokens per character,
    and fertility on different domains.
    """
    tokenizer = load_tokenizer(tokenizer_path)

    test_cases = {
        "english": "The quick brown fox jumps over the lazy dog. Natural language processing is a subfield of linguistics and artificial intelligence concerned with the interactions between computers and human language.",
        "math": "Calculate 12345 + 67890 = 80235. The derivative of x^2 is 2x. Solve: 3x + 7 = 22, so x = 5.",
        "code_python": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nresult = fibonacci(10)\nprint(f'Result: {result}')",
        "code_js": "const fetchData = async (url) => {\n  const response = await fetch(url);\n  const data = await response.json();\n  return data.map(item => item.name);\n};",
        "chinese": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "spanish": "La inteligencia artificial es la simulación de procesos de inteligencia humana por parte de máquinas, especialmente sistemas informáticos.",
        "mixed": "The model achieved 95.3% accuracy on MMLU benchmark. 模型在基准测试中表现优异。Code: print('hello world')",
    }

    results = {}
    for name, text in test_cases.items():
        encoded = tokenizer.encode(text)
        n_tokens = len(encoded.ids)
        n_words = len(text.split())
        n_chars = len(text)

        results[name] = {
            "tokens": n_tokens,
            "words": n_words,
            "chars": n_chars,
            "tokens_per_word": round(n_tokens / max(n_words, 1), 2),
            "tokens_per_char": round(n_tokens / max(n_chars, 1), 3),
        }

        # Check digit splitting
        if name == "math":
            digit_tokens = []
            for token_id, token_str in zip(encoded.ids, encoded.tokens):
                stripped = token_str.replace("Ġ", "").strip()
                if stripped.isdigit():
                    digit_tokens.append(stripped)
            results[name]["digit_tokens"] = digit_tokens[:20]

    return results
