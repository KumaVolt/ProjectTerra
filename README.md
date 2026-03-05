# ProjectTerra

A self-evolving multimodal LLM trained **entirely from scratch**. No fine-tuning of existing models — Terra builds its own transformer, tokenizer, and multimodal encoders from random initialization.

The goal: break records for small models while running on consumer hardware (MacBook Air M4 / iPhone).

## Architecture

Terra is a custom transformer with GQA, RoPE, SwiGLU, and RMSNorm. It scales from 150M to 3B parameters.

| Preset | Params | Use Case |
|--------|--------|----------|
| `terra_150m` | ~129M | Local training on MacBook Air M4 24GB |
| `terra_400m` | ~385M | Cloud GPU burst training |
| `terra_1b` | ~1.1B | Cloud GPU |
| `terra_3b` | ~2.6B | Final target |

### Multimodal Components

All modalities can be trained **in parallel** — each encoder/decoder is independent:

| Component | Module | Params | Function |
|-----------|--------|--------|----------|
| Text backbone | `TerraForCausalLM` | 129M–2.6B | Language understanding & generation |
| Vision encoder | `TerraVisionEncoder` | 3.8M–86M | Image understanding (ViT + projection) |
| Image generator | `TerraImageGenerator` | 10M–80M | Text-to-image (latent diffusion + VAE) |
| Audio encoder | `TerraAudioEncoder` | 5M–60M | Speech-to-text (Whisper-style) |
| Speech decoder | `TerraSpeechDecoder` | 3M–12M | Text-to-speech (neural audio codec) |

All encoders project into the same embedding space. The LLM processes a unified sequence:
```
[text] [<|image_start|>] [image_tokens] [<|image_end|>] [text] [<|audio_start|>] [audio_tokens] [<|audio_end|>] [text]
```

Image generation uses consistency distillation for **1-4 step generation** (24x faster than standard 50-step diffusion).

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/ProjectTerra.git
cd ProjectTerra
python -m venv .venv && source .venv/bin/activate
pip install -e ".[multimodal]"

# Set up API keys (at least one LLM required for evolution)
cp .env.example .env
# Edit .env with your keys

# Initialize: train tokenizer + download data + prepare chunks
terra init

# Check status
terra status
```

## Training

### Local Pre-training (MacBook)

```bash
# Pre-train on your Mac (auto steps based on data size)
terra pretrain

# Or specify steps
terra pretrain --max-steps 5000
```

### Cloud GPU Training

Cloud training creates a RunPod GPU pod, trains, uploads the checkpoint to HuggingFace, and self-destructs — no babysitting needed.

```bash
# See cost estimate first
terra cloud-train --estimate --gpu A100

# Run training (auto steps, async = safe to close laptop)
terra cloud-train --gpu A100 --async

# Check progress
terra cloud-status

# Download results when done
terra cloud-download
```

**Supported GPUs:** A100, H100, A10G, L4, T4, RTX4090

**Requirements:**
- `RUNPOD_API_KEY` — get one at [runpod.io](https://runpod.io)
- `HF_TOKEN` — for checkpoint upload to HuggingFace
- `HF_REPO_ID` — e.g. `YourName/terra-checkpoints`

### Download Multimodal Training Data

```bash
# Download all (minimal sample for testing)
terra download-multimodal all

# Full training sets
terra download-multimodal all --no-minimal

# Per modality
terra download-multimodal vision
terra download-multimodal image-gen
terra download-multimodal audio
terra download-multimodal tts
```

**Data sources:**

| Modality | Datasets | Content |
|----------|----------|---------|
| Text | fineweb_edu, openwebmath, the_stack, cosmopedia, slim_orca | Web text, math, code, textbooks, instructions |
| Vision | COCO Captions, CC3M | Image-caption pairs |
| Image Gen | LAION Art, DiffusionDB | Image-caption pairs (aesthetic) |
| Audio/STT | LibriSpeech, Common Voice | Audio-transcript pairs |
| TTS | LJ Speech, LibriTTS-R | High-quality speech-text pairs |

## Self-Evolution

Terra evolves itself every 8 hours via GitHub Actions:

1. **Triage** — scans open issues, recent sessions, decides what to work on
2. **Initialize** — trains tokenizer + downloads data (first run only)
3. **Train** — launches cloud GPU for pre-training
4. **Evolve** — research, data generation, fine-tuning, evaluation
5. **Log** — commits results, comments on addressed issues

```bash
# Run one evolution session manually
terra evolve

# Or trigger via GitHub Actions
gh workflow run evolve.yml
```

### Manual Trigger Options

```bash
# Full cycle
gh workflow run evolve.yml -f mode=full

# Only research (no training)
gh workflow run evolve.yml -f mode=research-only

# Train with specific GPU
gh workflow run evolve.yml -f mode=train-only -f gpu=H100

# Fix a specific issue
gh workflow run evolve.yml -f mode=fix-issue -f target_issue=42

# Dry run (no commits)
gh workflow run evolve.yml -f dry_run=true
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `terra init` | Initialize pipeline (tokenizer + data) |
| `terra status` | Show current state |
| `terra pretrain` | Pre-train locally |
| `terra cloud-train` | Pre-train on cloud GPU |
| `terra cloud-status` | Check cloud job progress |
| `terra cloud-download` | Download cloud results |
| `terra train-tokenizer` | Train BPE tokenizer |
| `terra download-data` | Download text pre-training data |
| `terra prepare-data` | Tokenize and chunk data |
| `terra download-multimodal` | Download multimodal training data |
| `terra evaluate` | Run benchmarks |
| `terra generate-data` | Generate synthetic training data |
| `terra serve` | Start local inference server |
| `terra evolve` | Run one evolution session |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | One of these | Claude API key for research/distillation |
| `GLM_API_KEY` | One of these | GLM API key (alternative to Claude) |
| `RUNPOD_API_KEY` | For cloud training | RunPod GPU API key |
| `HF_TOKEN` | For cloud training | HuggingFace token |
| `HF_REPO_ID` | For cloud training | HuggingFace checkpoint repo |

## Project Structure

```
ProjectTerra/
├── configs/
│   ├── terra.yaml              # Main config (architecture, training, data)
│   └── evolution_state.json    # Current evolution state
├── src/
│   ├── core/
│   │   ├── cli.py              # CLI entrypoint
│   │   ├── orchestrator.py     # Evolution pipeline
│   │   ├── llm_client.py       # Claude + GLM API clients
│   │   └── session_logger.py   # Logging + git commit
│   ├── training/
│   │   ├── model.py            # Custom transformer (GQA, RoPE, SwiGLU)
│   │   ├── pretrain.py         # Pre-training loop (early stopping, auto steps)
│   │   ├── tokenizer.py        # BPE tokenizer training
│   │   ├── cloud.py            # RunPod cloud GPU integration
│   │   ├── vision_encoder.py   # ViT image encoder
│   │   ├── image_generator.py  # Latent diffusion (text-to-image)
│   │   ├── audio_encoder.py    # Whisper-style STT encoder
│   │   ├── speech_decoder.py   # Neural codec TTS decoder
│   │   └── multimodal.py       # Unified multimodal wrapper
│   ├── data/
│   │   ├── downloader.py       # Text pre-training data
│   │   ├── multimodal_downloader.py  # Vision/audio/TTS data
│   │   └── generator.py        # Synthetic data generation
│   └── evaluation/
│       └── benchmarks.py       # lm-eval benchmark runner
├── scripts/
│   ├── cloud_train.py          # Standalone script for cloud pods
│   ├── triage.py               # GitHub Actions triage
│   └── run_evolution.py        # GitHub Actions evolution
├── .github/workflows/
│   └── evolve.yml              # 8-hour evolution cron
└── models/
    └── tokenizer/              # Trained BPE tokenizer (32K vocab)
```

## How It Works

### From-Scratch Training

Unlike most LLM projects that fine-tune existing models, Terra:
1. Designs its own transformer architecture
2. Trains its own BPE tokenizer on the training data
3. Pre-trains from random weight initialization
4. Builds its own vision/audio/speech encoders

### Self-Hosting

Once Terra's benchmark score exceeds the threshold (default: 0.65), it starts using **itself** for research, evaluation, and data generation — alongside (or instead of) external LLMs like Claude.

```bash
# Start Terra's own inference server
terra serve models/current --port 8080

# Set in .env to use Terra for evolution
TERRA_SERVER_URL=http://localhost:8080
```

### Cost

Cloud pre-training on an A100 costs ~$1-3 per session with auto steps and early stopping. The model self-destructs the GPU pod when done.

## License

MIT
