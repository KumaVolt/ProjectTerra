"""Benchmark evaluation system for tracking model improvements."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class BenchmarkRunner:
    """Runs standardized benchmarks against the model."""

    SUPPORTED_BENCHMARKS = {
        "mmlu": {"task": "mmlu", "metric": "acc"},
        "arc_challenge": {"task": "arc_challenge", "metric": "acc_norm"},
        "hellaswag": {"task": "hellaswag", "metric": "acc_norm"},
        "gsm8k": {"task": "gsm8k", "metric": "exact_match"},
        "humaneval": {"task": "humaneval", "metric": "pass@1"},
        "truthfulqa": {"task": "truthfulqa_mc2", "metric": "acc"},
    }

    def __init__(self, config: dict):
        self.config = config
        self.results_dir = Path("logs/benchmarks")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_config = config.get("evaluation", {})

    def run_benchmark(
        self, model_path: str, benchmark: str, num_samples: int | None = None
    ) -> dict:
        """Run a single benchmark using lm-evaluation-harness."""
        if benchmark not in self.SUPPORTED_BENCHMARKS:
            return {"error": f"Unknown benchmark: {benchmark}"}

        bench_info = self.SUPPORTED_BENCHMARKS[benchmark]
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True",
            "--tasks", bench_info["task"],
            "--batch_size", "auto",
            "--output_path", str(self.results_dir / f"{benchmark}_latest"),
        ]

        if num_samples:
            cmd.extend(["--limit", str(num_samples)])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            return self._parse_results(benchmark, result.stdout)
        except subprocess.TimeoutExpired:
            return {"error": "Benchmark timed out", "benchmark": benchmark}
        except Exception as e:
            return {"error": str(e), "benchmark": benchmark}

    def run_quick_eval(self, model_path: str) -> dict:
        """Run a quick evaluation with limited samples for fast feedback."""
        num_samples = self.eval_config.get("quick_eval_samples", 100)
        benchmarks = self.eval_config.get("benchmarks", [])
        results = {}

        for bench in benchmarks:
            name = bench["name"]
            result = self.run_benchmark(model_path, name, num_samples)
            results[name] = result

        return results

    def run_full_eval(self, model_path: str) -> dict:
        """Run full evaluation on all benchmarks."""
        benchmarks = self.eval_config.get("benchmarks", [])
        results = {}
        for bench in benchmarks:
            name = bench["name"]
            results[name] = self.run_benchmark(model_path, name)
        return results

    def compute_weighted_score(self, results: dict) -> float:
        """Compute a weighted aggregate score from benchmark results."""
        benchmarks = {b["name"]: b["weight"] for b in self.eval_config.get("benchmarks", [])}
        total_weight = 0
        weighted_sum = 0

        for name, weight in benchmarks.items():
            if name in results and "score" in results[name]:
                weighted_sum += results[name]["score"] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def is_improvement(self, old_results: dict, new_results: dict) -> bool:
        """Check if new results represent a meaningful improvement."""
        old_score = self.compute_weighted_score(old_results)
        new_score = self.compute_weighted_score(new_results)
        min_improvement = self.eval_config.get("min_improvement", 0.005)
        return (new_score - old_score) >= min_improvement

    def save_results(self, results: dict, label: str):
        """Save benchmark results with timestamp."""
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "results": results,
            "weighted_score": self.compute_weighted_score(results),
        }
        path = self.results_dir / f"eval_{label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(output, indent=2))
        return path

    def get_latest_results(self) -> dict | None:
        """Load the most recent benchmark results."""
        files = sorted(self.results_dir.glob("eval_*.json"), reverse=True)
        if files:
            return json.loads(files[0].read_text())
        return None

    def _parse_results(self, benchmark: str, output: str) -> dict:
        """Parse lm-eval output to extract scores."""
        bench_info = self.SUPPORTED_BENCHMARKS[benchmark]
        metric = bench_info["metric"]

        # Try to find the JSON results file
        results_path = self.results_dir / f"{benchmark}_latest"
        results_files = list(results_path.glob("**/*.json")) if results_path.exists() else []
        for rf in results_files:
            try:
                data = json.loads(rf.read_text())
                if "results" in data:
                    for task_name, task_results in data["results"].items():
                        if metric in task_results:
                            return {
                                "benchmark": benchmark,
                                "score": task_results[metric],
                                "metric": metric,
                                "raw": task_results,
                            }
            except (json.JSONDecodeError, KeyError):
                continue

        # Fallback: parse stdout
        for line in output.split("\n"):
            if metric in line and benchmark in line.lower():
                parts = line.split("|")
                for part in parts:
                    part = part.strip()
                    try:
                        score = float(part)
                        if 0 <= score <= 1:
                            return {"benchmark": benchmark, "score": score, "metric": metric}
                    except ValueError:
                        continue

        return {"benchmark": benchmark, "error": "Could not parse results", "raw_output": output[:500]}
