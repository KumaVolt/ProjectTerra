"""Training module for Terra: supports both from-scratch pre-training and QLoRA fine-tuning."""

import json
from pathlib import Path

import torch


class TerraTrainer:
    """Handles both pre-training from scratch and QLoRA fine-tuning."""

    def __init__(self, config: dict):
        self.config = config
        self.training_config = config["training"]
        self.output_dir = Path("models/checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data: list[dict], run_name: str = "evolution") -> dict:
        """Run training - routes to pre-train or fine-tune based on config/state."""
        model_path = self._get_current_model_path()

        if model_path and Path(model_path).exists():
            # We have a pre-trained model - fine-tune it
            return self.finetune(data, model_path, run_name)
        else:
            # No model yet - need pre-training first
            return self.pretrain_session(run_name)

    def pretrain_session(self, run_name: str = "pretrain") -> dict:
        """Run a pre-training session (or continue one)."""
        from src.training.pretrain import pretrain

        arch_config = self.config.get("architecture", {})
        pretrain_config = self.training_config.get("pretrain", {})

        # Check for existing checkpoint to resume
        resume_from = None
        ckpt_dir = self.output_dir / "pretrain"
        if ckpt_dir.exists():
            checkpoints = sorted(ckpt_dir.glob("checkpoint-*/checkpoint.pt"))
            if checkpoints:
                resume_from = str(checkpoints[-1])
                print(f"Resuming pre-training from {resume_from}")

        result = pretrain(
            model_config=arch_config,
            data_path=pretrain_config.get("data_path", "data/pretrain_chunks"),
            output_dir=str(self.output_dir / "pretrain"),
            batch_size=pretrain_config.get("batch_size", 4),
            gradient_accumulation_steps=pretrain_config.get("gradient_accumulation_steps", 8),
            learning_rate=pretrain_config.get("learning_rate", 3e-4),
            max_steps=pretrain_config.get("max_steps_per_session", 2000),
            warmup_steps=pretrain_config.get("warmup_steps", 200),
            save_steps=pretrain_config.get("save_steps", 500),
            use_gradient_checkpointing=pretrain_config.get("gradient_checkpointing", True),
            resume_from=resume_from,
        )

        # Link as current model if training completed
        if "model_path" in result:
            current = Path("models/current")
            current.mkdir(parents=True, exist_ok=True)
            # Copy config and model files
            import shutil
            src = Path(result["model_path"])
            if src.exists():
                for f in src.iterdir():
                    shutil.copy2(f, current / f.name)

        return result

    def finetune(self, data: list[dict], model_path: str, run_name: str = "finetune") -> dict:
        """QLoRA fine-tune an existing model on instruction data."""
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer

        # Try loading as Terra model first, fall back to HF model
        model, tokenizer = self._load_model_for_finetune(model_path)

        lora_config = LoraConfig(
            r=self.training_config.get("lora_r", 16),
            lora_alpha=self.training_config.get("lora_alpha", 32),
            lora_dropout=self.training_config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        # Format data for chat
        formatted = []
        for item in data:
            text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
            formatted.append({"text": text})
        dataset = Dataset.from_list(formatted)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir / run_name),
            num_train_epochs=1,
            per_device_train_batch_size=self.training_config.get("batch_size", 4),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.training_config.get("learning_rate", 2e-4),
            max_steps=min(self.training_config.get("max_steps", 500), len(dataset)),
            warmup_steps=self.training_config.get("warmup_steps", 50),
            logging_steps=10,
            save_steps=self.training_config.get("save_steps", 100),
            save_total_limit=3,
            bf16=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=self.config["model"].get("context_length", 8192),
        )

        result = trainer.train()

        adapter_path = self.output_dir / run_name / "final_adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        return {
            "train_loss": result.training_loss,
            "steps": result.global_step,
            "adapter_path": str(adapter_path),
            "samples_trained": len(dataset),
            "method": "qlora",
        }

    def _load_model_for_finetune(self, model_path: str):
        """Load model for fine-tuning. Supports both Terra and HF models."""
        config_path = Path(model_path) / "config.json"

        if config_path.exists():
            config_data = json.loads(config_path.read_text())
            # Check if it's a Terra model (has our specific fields)
            if "num_key_value_heads" in config_data and "rope_theta" in config_data:
                return self._load_terra_for_finetune(model_path)

        # Fall back to HF transformers loading
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
        return model, tokenizer

    def _load_terra_for_finetune(self, model_path: str):
        """Load a Terra model for fine-tuning with LoRA."""
        from src.training.model import TerraForCausalLM
        from src.training.tokenizer import load_tokenizer

        model = TerraForCausalLM.from_pretrained(model_path)
        tokenizer = load_tokenizer(
            self.config.get("architecture", {}).get("tokenizer_path", "models/tokenizer")
        )
        return model, tokenizer

    def _get_current_model_path(self) -> str | None:
        """Get path to current best model."""
        current = Path("models/current")
        if current.exists() and (current / "config.json").exists():
            return str(current)

        # Check evolution state
        state_path = Path("configs/evolution_state.json")
        if state_path.exists():
            state = json.loads(state_path.read_text())
            return state.get("best_model_path")

        return None
