"""Synthetic data generation for model training."""

import json
import random
from pathlib import Path

from src.core.llm_client import LLMPool


class DataGenerator:
    """Generates high-quality training data using external LLMs."""

    DOMAINS = [
        "reasoning",
        "coding",
        "math",
        "science",
        "creative_writing",
        "multilingual",
        "instruction_following",
        "common_sense",
        "conversation",
        "summarization",
    ]

    COMPLEXITY_PROMPTS = {
        1: "Generate simple, straightforward examples suitable for beginners.",
        2: "Generate intermediate examples that require some reasoning.",
        3: "Generate advanced examples that require multi-step reasoning.",
        4: "Generate expert-level examples with complex reasoning chains.",
        5: "Generate research-level examples that push the boundaries of understanding.",
    }

    def __init__(self, config: dict, llm_pool: LLMPool):
        self.config = config
        self.llm_pool = llm_pool
        self.data_dir = Path("data/generated")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_config = config.get("data", {})

    def generate_distillation_data(
        self, domain: str, num_samples: int = 50
    ) -> list[dict]:
        """Generate training data via knowledge distillation from teacher models."""
        all_data = self.llm_pool.generate_data(domain, num_samples)
        return self._filter_quality(all_data)

    def generate_evol_instruct_data(
        self, seed_instructions: list[str], complexity: int = 3
    ) -> list[dict]:
        """Generate data using Evol-Instruct methodology."""
        complexity_note = self.COMPLEXITY_PROMPTS.get(complexity, "")
        evolved = []

        client = self.llm_pool.get_any()
        if not client:
            return []

        for instruction in seed_instructions:
            prompt = f"""Evolve the following instruction to make it more complex and challenging.
{complexity_note}

Original instruction: {instruction}

Create 3 evolved versions of increasing complexity. For each, provide:
1. The evolved instruction
2. A detailed, high-quality response

Return as a JSON array with "instruction" and "response" fields. Only JSON."""

            try:
                resp = client.query(prompt, max_tokens=4096, temperature=0.8)
                text = resp.content
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    items = json.loads(text[start:end])
                    for item in items:
                        item["source"] = "evol_instruct"
                        item["complexity"] = complexity
                    evolved.extend(items)
            except (json.JSONDecodeError, Exception):
                continue

        return self._filter_quality(evolved)

    def generate_self_instruct_data(self, num_samples: int = 50) -> list[dict]:
        """Generate data using Self-Instruct methodology."""
        seed_path = Path("configs/seed_tasks.json")
        if seed_path.exists():
            seeds = json.loads(seed_path.read_text())
        else:
            seeds = self._default_seed_tasks()

        client = self.llm_pool.get_any()
        if not client:
            return []

        seed_sample = random.sample(seeds, min(5, len(seeds)))
        seed_text = "\n".join(f"- {s['instruction']}" for s in seed_sample)

        prompt = f"""Given these example instructions:
{seed_text}

Generate {num_samples} new, diverse instructions with responses. The instructions should:
1. Cover different topics and skills
2. Range from simple to complex
3. Include some that require reasoning, coding, math, or creativity
4. Be self-contained

Return as a JSON array with "instruction" and "response" fields. Only JSON."""

        try:
            resp = client.query(prompt, max_tokens=8192, temperature=0.9)
            text = resp.content
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(text[start:end])
                for item in items:
                    item["source"] = "self_instruct"
                return self._filter_quality(items)
        except (json.JSONDecodeError, Exception):
            pass
        return []

    def generate_multimodal_data(self, modality: str, num_samples: int = 20) -> list[dict]:
        """Generate training data for multimodal capabilities."""
        client = self.llm_pool.get_any()
        if not client:
            return []

        modality_prompts = {
            "vision": "Generate image understanding tasks: describe what you'd see, visual QA, chart reading, etc. Include the expected description/answer.",
            "speech_to_text": "Generate speech transcription training pairs: include various accents, speeds, and speaking styles. Describe the audio and provide the transcription.",
            "text_to_speech": "Generate text-to-speech training pairs: include prosody instructions, emotion markers, and the text to be spoken.",
        }

        prompt = f"""{modality_prompts.get(modality, 'Generate training data.')}

Generate {num_samples} examples as a JSON array with "instruction" and "response" fields. Only JSON."""

        try:
            resp = client.query(prompt, max_tokens=8192, temperature=0.8)
            text = resp.content
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(text[start:end])
                for item in items:
                    item["source"] = f"multimodal_{modality}"
                return items
        except (json.JSONDecodeError, Exception):
            pass
        return []

    def generate_session_data(self, focus_domains: list[str] | None = None) -> list[dict]:
        """Generate a full session's worth of training data."""
        domains = focus_domains or random.sample(self.DOMAINS, min(3, len(self.DOMAINS)))
        max_samples = self.data_config.get("max_samples_per_session", 1000)
        samples_per_domain = max_samples // len(domains)

        all_data = []
        for domain in domains:
            data = self.generate_distillation_data(domain, samples_per_domain)
            all_data.extend(data)

        # Add some self-instruct data
        si_data = self.generate_self_instruct_data(min(50, max_samples - len(all_data)))
        all_data.extend(si_data)

        # Add multimodal data if configured
        modalities = self.config.get("model", {}).get("modalities", [])
        for modality in modalities:
            if modality != "text":
                mm_data = self.generate_multimodal_data(modality, 20)
                all_data.extend(mm_data)

        random.shuffle(all_data)
        return all_data[:max_samples]

    def save_data(self, data: list[dict], label: str) -> Path:
        """Save generated data to disk."""
        path = self.data_dir / f"{label}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def _filter_quality(self, data: list[dict]) -> list[dict]:
        """Filter data by basic quality heuristics."""
        min_score = self.data_config.get("min_quality_score", 0.7)
        filtered = []
        for item in data:
            if not item.get("instruction") or not item.get("response"):
                continue
            if len(item["instruction"]) < 10:
                continue
            if len(item["response"]) < 20:
                continue
            # Use LLM judge for a sample if available
            filtered.append(item)
        return filtered

    def _default_seed_tasks(self) -> list[dict]:
        return [
            {"instruction": "Explain the concept of recursion in programming with an example."},
            {"instruction": "Write a Python function to find the longest common subsequence of two strings."},
            {"instruction": "What are the key differences between TCP and UDP protocols?"},
            {"instruction": "Solve: If a train travels 120 km in 2 hours, what is its average speed in m/s?"},
            {"instruction": "Translate 'The weather is beautiful today' into Spanish, French, and German."},
            {"instruction": "Explain how photosynthesis works at the molecular level."},
            {"instruction": "Write a haiku about artificial intelligence."},
            {"instruction": "Debug this code: def factorial(n): return n * factorial(n)"},
            {"instruction": "What are the ethical implications of self-improving AI systems?"},
            {"instruction": "Explain the difference between supervised and unsupervised learning."},
        ]
