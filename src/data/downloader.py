"""Download and prepare pre-training datasets.

Uses streaming where possible to avoid filling disk.
Mixes multiple data sources for diversity and to prevent model collapse.
"""

import json
from pathlib import Path

from datasets import load_dataset


# Pre-training data sources with mixing ratios
DATA_SOURCES = {
    "fineweb_edu": {
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "hf_subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "mix_ratio": 0.40,
        "description": "High-quality educational web text",
    },
    "openwebmath": {
        "hf_name": "open-web-math/open-web-math",
        "hf_subset": None,
        "split": "train",
        "text_field": "text",
        "mix_ratio": 0.15,
        "description": "Mathematical web pages",
    },
    "the_stack_smol": {
        "hf_name": "bigcode/the-stack-smol",
        "hf_subset": "data",
        "split": "train",
        "text_field": "content",
        "mix_ratio": 0.20,
        "description": "Curated code from The Stack",
    },
    "cosmopedia": {
        "hf_name": "HuggingFaceTB/cosmopedia",
        "hf_subset": None,
        "split": "train",
        "text_field": "text",
        "mix_ratio": 0.15,
        "description": "Synthetic textbook-quality data",
    },
    "slim_orca": {
        "hf_name": "Open-Orca/SlimOrca",
        "hf_subset": None,
        "split": "train",
        "text_field": None,  # instruction/response format
        "mix_ratio": 0.10,
        "description": "Instruction-following data",
    },
}


def download_pretraining_data(
    output_dir: str = "data/pretrain",
    max_samples_per_source: int = 50000,
    sources: list[str] | None = None,
) -> dict[str, str]:
    """Download pre-training data to disk as text files.

    Args:
        output_dir: Where to save the data.
        max_samples_per_source: Max samples per source.
        sources: Subset of DATA_SOURCES to use. None = all.

    Returns:
        Dict mapping source name to output file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {}

    source_keys = sources or list(DATA_SOURCES.keys())

    for source_name in source_keys:
        if source_name not in DATA_SOURCES:
            print(f"[download] Unknown source: {source_name}, skipping")
            continue

        info = DATA_SOURCES[source_name]
        output_file = out / f"{source_name}.txt"

        if output_file.exists():
            print(f"[download] {source_name} already exists at {output_file}, skipping")
            result[source_name] = str(output_file)
            continue

        print(f"[download] Fetching {source_name}: {info['description']}...")

        try:
            kwargs = {"path": info["hf_name"], "split": info["split"], "streaming": True}
            if info["hf_subset"]:
                kwargs["name"] = info["hf_subset"]

            ds = load_dataset(**kwargs)

            count = 0
            with open(output_file, "w", encoding="utf-8") as f:
                for example in ds:
                    if count >= max_samples_per_source:
                        break

                    text = _extract_text(example, info)
                    if text and len(text) > 100:  # Skip very short texts
                        f.write(text.strip() + "\n\n")
                        count += 1

                    if count % 10000 == 0 and count > 0:
                        print(f"  [{source_name}] {count}/{max_samples_per_source}")

            print(f"[download] {source_name}: saved {count} samples to {output_file}")
            result[source_name] = str(output_file)

        except Exception as e:
            print(f"[download] {source_name} failed: {e}")
            if output_file.exists():
                output_file.unlink()

    return result


def _extract_text(example: dict, source_info: dict) -> str:
    """Extract text from a dataset example based on source format."""
    text_field = source_info.get("text_field")

    if text_field:
        return example.get(text_field, "")

    # Instruction-response format (SlimOrca, etc.)
    parts = []
    if "system_prompt" in example and example["system_prompt"]:
        parts.append(f"<|system|>\n{example['system_prompt']}")
    if "question" in example:
        parts.append(f"<|user|>\n{example['question']}")
    elif "instruction" in example:
        parts.append(f"<|user|>\n{example['instruction']}")
    if "response" in example:
        parts.append(f"<|assistant|>\n{example['response']}")
    elif "output" in example:
        parts.append(f"<|assistant|>\n{example['output']}")

    return "\n".join(parts)


def prepare_pretraining_chunks(
    data_dir: str = "data/pretrain",
    output_dir: str = "data/pretrain_chunks",
    chunk_size: int = 2048,
    tokenizer_path: str = "models/tokenizer",
) -> str:
    """Tokenize and chunk pre-training data for efficient training.

    Concatenates all text, tokenizes, and splits into fixed-length chunks.
    This is the standard approach for pre-training (no padding waste).

    Args:
        data_dir: Directory with raw text files.
        output_dir: Where to save chunked data.
        chunk_size: Sequence length for each training example.
        tokenizer_path: Path to trained tokenizer.

    Returns:
        Path to output directory with chunked data.
    """
    from src.training.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)
    data_path = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect all text files and their mix ratios
    text_files = []
    for source_name, info in DATA_SOURCES.items():
        fpath = data_path / f"{source_name}.txt"
        if fpath.exists():
            text_files.append((fpath, info["mix_ratio"]))

    if not text_files:
        raise FileNotFoundError(f"No pre-training data found in {data_dir}")

    # Process each source
    all_token_ids = []
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    for fpath, ratio in text_files:
        print(f"[chunk] Tokenizing {fpath.name}...")
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                encoded = tokenizer.encode(line)
                if bos_id is not None:
                    all_token_ids.append(bos_id)
                all_token_ids.extend(encoded.ids)
                if eos_id is not None:
                    all_token_ids.append(eos_id)

    # Split into chunks
    num_chunks = len(all_token_ids) // chunk_size
    print(f"[chunk] Total tokens: {len(all_token_ids):,}, chunks: {num_chunks:,}")

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        chunk = all_token_ids[start : start + chunk_size]
        chunks.append(chunk)

    # Save as JSONL (each line is a chunk of token IDs)
    output_file = out / "train.jsonl"
    with open(output_file, "w") as f:
        for chunk in chunks:
            f.write(json.dumps({"input_ids": chunk}) + "\n")

    print(f"[chunk] Saved {len(chunks)} chunks to {output_file}")
    return str(out)


def download_minimal_sample(
    output_dir: str = "data/pretrain",
    num_samples: int = 5000,
) -> dict[str, str]:
    """Download a minimal sample for local testing on MacBook."""
    return download_pretraining_data(
        output_dir=output_dir,
        max_samples_per_source=num_samples,
        sources=["fineweb_edu", "the_stack_smol"],
    )
