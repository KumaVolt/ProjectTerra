"""Main orchestrator for the self-evolution pipeline."""

import json
import traceback
from pathlib import Path

import yaml

from src.core.issue_tracker import IssueTracker
from src.core.llm_client import LLMPool
from src.core.session_logger import SessionLogger
from src.data.generator import DataGenerator
from src.evaluation.benchmarks import BenchmarkRunner
from src.training.trainer import TerraTrainer


class EvolutionOrchestrator:
    """Manages the full self-evolution cycle."""

    def __init__(self, config_path: str = "configs/terra.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.logger = SessionLogger(
            log_dir=self.config.get("logging", {}).get("session_log_dir", "logs/sessions"),
            repo_root=".",
        )
        self.llm_pool = LLMPool()
        self.issue_tracker = IssueTracker()
        self.data_generator = DataGenerator(self.config, self.llm_pool)
        self.benchmark_runner = BenchmarkRunner(self.config)
        self.trainer = TerraTrainer(self.config)
        self.state = self._load_state()

    def run(self):
        """Execute one evolution session."""
        self.logger.log("init", f"Starting evolution session {self.logger.session_id}")
        self.logger.log("init", f"Available LLMs: {self.llm_pool.available()}")
        self.logger.log("init", f"Training stage: {self.state.get('training_stage', 'pretrain')}")

        try:
            # Stage 1: Check issues
            plan = self._check_issues_and_plan()
            self.logger.log_stage_complete("check_issues", plan)

            # Stage 2: Research
            research = self._research(plan)
            self.logger.log_stage_complete("research", {"findings_count": len(research)})

            # Route based on training stage
            training_stage = self.state.get("training_stage", "pretrain")
            train_result = None

            if training_stage == "pretrain":
                # Pre-training from scratch
                train_result = self._pretrain_session()
                self.logger.log_stage_complete("pretrain", train_result)

                # Check if pre-training is far enough to move to fine-tuning
                if train_result and train_result.get("total_steps", 0) > 0:
                    self.state["pretrain_steps_completed"] = (
                        self.state.get("pretrain_steps_completed", 0)
                        + train_result.get("total_steps", 0)
                    )
            else:
                # Fine-tuning stage: generate data and fine-tune
                data = self._generate_data(plan, research)
                self.logger.log_stage_complete("generate_data", {"samples": len(data)})

                if data and len(data) >= 10:
                    train_result = self._train(data)
                    self.logger.log_stage_complete("train", train_result)
                else:
                    self.logger.log("train", "Skipping training: insufficient data")

            # Evaluate if we have a model
            model_path = self._get_eval_model_path(train_result)
            if model_path:
                eval_result = self._evaluate(model_path)
                self.logger.log_stage_complete("evaluate", eval_result)
            else:
                self.logger.log("evaluate", "Skipping evaluation: no model to evaluate")

            # Stage 6: Log and commit
            self.logger.set_token_usage(self.llm_pool.get_token_usage())
            self.logger.finalize("completed")

        except Exception as e:
            self.logger.log_error("pipeline", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            self.logger.finalize("failed")

        # Always commit the session log
        self._save_state()
        self.logger.commit_to_repo()

    def _check_issues_and_plan(self) -> dict:
        """Check GitHub issues and plan the session."""
        self.logger.log("check_issues", "Checking GitHub issues...")
        issues = self.issue_tracker.get_actionable_issues()
        self.logger.log("check_issues", f"Found {len(issues)} actionable issues")

        plan = {
            "issues": [],
            "focus_areas": [],
            "strategy": "research_and_improve",
        }

        if issues:
            # Pick the highest priority issues
            for issue in issues[:3]:
                category = self.issue_tracker.categorize_issue(issue)
                plan["issues"].append({
                    "number": issue.number,
                    "title": issue.title,
                    "category": category,
                })
                if category not in plan["focus_areas"]:
                    plan["focus_areas"].append(category)

            self.logger.log("check_issues", f"Focus areas: {plan['focus_areas']}")
        else:
            # No issues - default to general improvement
            plan["focus_areas"] = self._identify_weak_areas()
            plan["strategy"] = "autonomous_improvement"
            self.logger.log("check_issues", "No issues found, pursuing autonomous improvement")

        # Consult LLMs for planning
        if self.llm_pool.available():
            recent_sessions = self.logger.get_recent_sessions(3)
            session_summary = json.dumps(
                [{"id": s["session_id"], "stages": s["stages_completed"],
                  "improvements": s.get("improvements", {})}
                 for s in recent_sessions],
                indent=2,
            )

            planning_prompt = f"""You are planning an evolution session for ProjectTerra, a self-evolving small LLM.

Recent sessions: {session_summary}

Current focus areas: {plan['focus_areas']}
Open issues: {json.dumps(plan['issues'], indent=2)}

Current model: terra (custom architecture, trained from scratch)
Architecture: {self.config.get('architecture', {}).get('preset', 'terra_150m')}
Target: Run on iPhone 15 / MacBook Air while maximizing benchmark scores.
Modalities: text, vision, speech-to-text, text-to-speech, full-duplex

What specific improvements should this session focus on? Consider:
1. What training data domains would be most impactful?
2. What techniques could improve the model most?
3. Any architectural changes to consider?

Be specific and actionable. Return a JSON object with:
- "data_domains": list of domains to generate data for
- "techniques": list of specific techniques to try
- "rationale": brief explanation"""

            responses = self.llm_pool.research(planning_prompt)
            for source, response in responses.items():
                try:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start >= 0 and end > start:
                        llm_plan = json.loads(response[start:end])
                        plan["llm_recommendations"] = {source: llm_plan}
                        if "data_domains" in llm_plan:
                            plan["data_domains"] = llm_plan["data_domains"]
                except (json.JSONDecodeError, Exception):
                    pass

        if "data_domains" not in plan:
            plan["data_domains"] = ["reasoning", "coding", "math"]

        return plan

    def _research(self, plan: dict) -> list[dict]:
        """Conduct research using external LLMs."""
        self.logger.log("research", "Conducting research...")
        findings = []

        research_questions = [
            f"What are the latest techniques to improve a small custom transformer model trained from scratch on {', '.join(plan.get('data_domains', ['general']))} tasks?",
            "What recent papers or techniques could help a 150M-3B parameter model trained from scratch achieve state-of-the-art performance on standard benchmarks?",
        ]

        # Add issue-specific research
        for issue in plan.get("issues", []):
            research_questions.append(
                f"Research approach for: {issue['title']} (category: {issue['category']})"
            )

        for question in research_questions:
            responses = self.llm_pool.research(question)
            findings.append({"question": question, "responses": responses})
            self.logger.log("research", f"Researched: {question[:80]}...")

        return findings

    def _generate_data(self, plan: dict, research: list[dict]) -> list[dict]:
        """Generate training data based on the plan."""
        self.logger.log("generate_data", "Generating training data...")
        domains = plan.get("data_domains", ["reasoning", "coding", "math"])
        data = self.data_generator.generate_session_data(domains)
        self.logger.log("generate_data", f"Generated {len(data)} training samples")

        if data:
            path = self.data_generator.save_data(data, f"session_{self.logger.session_id}")
            self.logger.log("generate_data", f"Data saved to {path}")

        return data

    def _pretrain_session(self) -> dict:
        """Run a pre-training session from scratch."""
        self.logger.log("pretrain", "Running pre-training session...")
        try:
            result = self.trainer.pretrain_session(run_name=f"session_{self.logger.session_id}")
            loss = result.get("final_loss", "N/A")
            self.logger.log("pretrain", f"Pre-training session complete. Loss: {loss}")
            return result
        except Exception as e:
            self.logger.log_error("pretrain", str(e))
            return {"error": str(e)}

    def _train(self, data: list[dict]) -> dict:
        """Fine-tune the model on generated data."""
        self.logger.log("train", f"Fine-tuning on {len(data)} samples...")
        try:
            result = self.trainer.train(data, run_name=f"session_{self.logger.session_id}")
            loss = result.get("train_loss", "N/A")
            self.logger.log("train", f"Training complete. Loss: {loss}")
            return result
        except Exception as e:
            self.logger.log_error("train", str(e))
            return {"error": str(e)}

    def _get_eval_model_path(self, train_result: dict | None) -> str | None:
        """Get the model path to evaluate from training results."""
        if not train_result:
            return None
        # Pre-training saves to model_path
        if "model_path" in train_result:
            return train_result["model_path"]
        # Fine-tuning saves to adapter_path
        if "adapter_path" in train_result:
            return train_result["adapter_path"]
        return None

    def _evaluate(self, model_path: str) -> dict:
        """Evaluate the trained model."""
        self.logger.log("evaluate", "Running benchmarks...")
        try:
            results = self.benchmark_runner.run_quick_eval(model_path)
            self.benchmark_runner.save_results(results, f"session_{self.logger.session_id}")

            # Compare with previous best
            previous = self.benchmark_runner.get_latest_results()
            if previous and self.benchmark_runner.is_improvement(
                previous.get("results", {}), results
            ):
                self.logger.log("evaluate", "NEW BEST MODEL - improvement detected!")
                self.state["best_model_path"] = model_path
                self.state["best_score"] = self.benchmark_runner.compute_weighted_score(results)
            else:
                self.logger.log("evaluate", "No significant improvement over previous best")

            # Log individual benchmark results
            for bench, result in results.items():
                if "score" in result:
                    prev_score = (
                        previous.get("results", {}).get(bench, {}).get("score", 0)
                        if previous else 0
                    )
                    self.logger.log_improvement(bench, prev_score, result["score"])

            return results
        except Exception as e:
            self.logger.log_error("evaluate", str(e))
            return {"error": str(e)}

    def _identify_weak_areas(self) -> list[str]:
        """Identify areas where the model needs the most improvement."""
        latest = self.benchmark_runner.get_latest_results()
        if not latest:
            return ["reasoning", "coding", "math"]

        results = latest.get("results", {})
        # Sort benchmarks by score (ascending) to find weakest areas
        scored = [
            (name, data.get("score", 1.0))
            for name, data in results.items()
            if "score" in data
        ]
        scored.sort(key=lambda x: x[1])

        # Map benchmarks to training domains
        bench_to_domain = {
            "mmlu": "knowledge",
            "arc_challenge": "reasoning",
            "hellaswag": "common_sense",
            "gsm8k": "math",
            "humaneval": "coding",
            "truthfulqa": "truthfulness",
        }

        return [bench_to_domain.get(s[0], "general") for s in scored[:3]]

    def _load_state(self) -> dict:
        """Load evolution state from disk."""
        state_path = Path("configs/evolution_state.json")
        if state_path.exists():
            return json.loads(state_path.read_text())
        return {
            "generation": 0,
            "best_model_path": None,
            "best_score": 0,
            "total_sessions": 0,
            "total_training_samples": 0,
        }

    def _save_state(self):
        """Save evolution state to disk."""
        self.state["generation"] += 1
        self.state["total_sessions"] += 1
        Path("configs/evolution_state.json").write_text(json.dumps(self.state, indent=2))
