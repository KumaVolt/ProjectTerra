"""Multimodal capability planner for ProjectTerra.

Handles the strategy for adding vision, STT, TTS, and full-duplex capabilities
to the Terra model. All modalities can be trained in parallel with text pre-training.
"""

from dataclasses import dataclass


@dataclass
class ModalityPlan:
    name: str
    approach: str
    estimated_size_mb: int
    priority: int
    dependencies: list[str]
    module: str  # Actual implementation module
    status: str = "implemented"


MODALITY_PLANS = {
    "text": ModalityPlan(
        name="text",
        approach="Custom transformer (GQA, RoPE, SwiGLU, RMSNorm). "
        "Pre-training from scratch on fineweb_edu + openwebmath + the_stack + cosmopedia + slim_orca.",
        estimated_size_mb=520,  # ~129M params in fp32, ~260MB in fp16, ~130MB Q4
        priority=1,
        dependencies=[],
        module="src.training.model.TerraForCausalLM",
        status="training",
    ),
    "vision": ModalityPlan(
        name="vision",
        approach="ViT encoder (tiny=3.8M, small=23M, base=86M) + 2-layer MLP projection. "
        "Patch embedding → transformer blocks → project to Terra hidden_size. "
        "Can be pre-trained independently with contrastive loss (SigLIP-style). "
        "Image patches become tokens in the same embedding space as text.",
        estimated_size_mb=100,  # small preset in fp16
        priority=2,
        dependencies=[],  # Can train independently!
        module="src.training.vision_encoder.TerraVisionEncoder",
    ),
    "speech_to_text": ModalityPlan(
        name="speech_to_text",
        approach="Whisper-style conv stem + transformer encoder (tiny=5M, small=14M, base=60M). "
        "Mel spectrogram → conv downsampling (4x) → transformer → projection to Terra space. "
        "CTC head for standalone ASR pre-training. Then projection fine-tuned for LLM integration.",
        estimated_size_mb=60,  # small preset in fp16
        priority=2,
        dependencies=[],  # Can train independently!
        module="src.training.audio_encoder.TerraAudioEncoder",
    ),
    "text_to_speech": ModalityPlan(
        name="text_to_speech",
        approach="Neural audio codec: encoder + residual VQ (4 codebooks x 1024) + decoder. "
        "Pre-train codec on speech reconstruction. LLM generates discrete audio tokens, "
        "codec decoder converts to mel spectrogram. LM conditioner bridges LLM → audio.",
        estimated_size_mb=15,  # small preset
        priority=3,
        dependencies=["text"],  # Needs LLM hidden states for conditioning
        module="src.training.speech_decoder.TerraSpeechDecoder",
    ),
    "image_generation": ModalityPlan(
        name="image_generation",
        approach="Latent diffusion: VAE compresses images to small latent space, "
        "U-Net denoiser predicts noise conditioned on Terra LLM text embeddings via cross-attention. "
        "Consistency distillation enables 1-4 step generation (vs 50+ standard). "
        "Presets: tiny=10M (64x64), small=60M (256x256), base=80M (512x512).",
        estimated_size_mb=250,  # small preset in fp16
        priority=2,
        dependencies=[],  # VAE trains independently, U-Net needs text embeddings
        module="src.training.image_generator.TerraImageGenerator",
    ),
    "full_duplex": ModalityPlan(
        name="full_duplex",
        approach="Dual-stream: STT encoder runs continuously while TTS decoder generates. "
        "Interrupt detector monitors incoming audio for speech onset. "
        "Requires streaming STT + interruptible TTS + attention over new input mid-generation.",
        estimated_size_mb=20,
        priority=4,
        dependencies=["speech_to_text", "text_to_speech"],
        module="src.training.multimodal.TerraMultimodal",
        status="planned",
    ),
}


def get_total_estimated_size() -> int:
    return sum(p.estimated_size_mb for p in MODALITY_PLANS.values())


def get_deployment_plan(max_memory_mb: int = 4096) -> list[ModalityPlan]:
    plans = sorted(MODALITY_PLANS.values(), key=lambda p: p.priority)
    selected = []
    total = 0
    for plan in plans:
        if total + plan.estimated_size_mb <= max_memory_mb:
            selected.append(plan)
            total += plan.estimated_size_mb
    return selected


def get_parallel_training_plan() -> list[dict]:
    """What can be trained RIGHT NOW in parallel."""
    return [
        {
            "track": "Text backbone",
            "module": "src.training.pretrain",
            "data": "fineweb_edu, openwebmath, the_stack_smol, cosmopedia, slim_orca",
            "hardware": "Cloud GPU (A100) or local Mac",
            "status": "ready",
        },
        {
            "track": "Vision encoder",
            "module": "src.training.vision_encoder",
            "data": "LAION-400M subset, CC3M, or SBU Captions (image-text pairs)",
            "hardware": "Can train on Mac (tiny preset) or cloud GPU (small/base)",
            "status": "ready — module implemented",
        },
        {
            "track": "Audio encoder (STT)",
            "module": "src.training.audio_encoder",
            "data": "LibriSpeech, Common Voice, GigaSpeech",
            "hardware": "Can train on Mac (tiny) or cloud GPU (small/base)",
            "status": "ready — module implemented, CTC pre-training supported",
        },
        {
            "track": "Image generator",
            "module": "src.training.image_generator",
            "data": "LAION-Aesthetics, CC3M, COCO (image-caption pairs)",
            "hardware": "Cloud GPU for U-Net training; VAE can train on Mac (tiny)",
            "status": "ready — VAE + U-Net + consistency distillation implemented",
            "fast_mode": "1-4 step generation via consistency distillation (10-24x speedup)",
        },
        {
            "track": "Speech codec (TTS)",
            "module": "src.training.speech_decoder",
            "data": "LibriTTS, VCTK, LJSpeech",
            "hardware": "Cloud GPU recommended",
            "status": "ready — codec pre-training supported",
        },
    ]
