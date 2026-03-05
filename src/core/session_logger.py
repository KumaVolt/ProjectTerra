"""Session logging system for evolution runs."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class SessionLogger:
    """Logs each evolution session and commits to the repository."""

    def __init__(self, log_dir: str = "logs/sessions", repo_root: str = "."):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{self.session_id}.json"
        self.entries: list[dict] = []
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "status": "running",
            "stages_completed": [],
            "improvements": {},
            "token_usage": {},
            "errors": [],
            "entries": self.entries,
        }

    def log(self, stage: str, message: str, data: dict | None = None):
        """Log an event during the session."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "message": message,
        }
        if data:
            entry["data"] = data
        self.entries.append(entry)
        print(f"[{stage}] {message}")

    def log_stage_complete(self, stage: str, result: dict | None = None):
        """Mark a stage as completed."""
        self.session_data["stages_completed"].append(stage)
        self.log(stage, f"Stage completed: {stage}", result)

    def log_error(self, stage: str, error: str):
        """Log an error."""
        self.session_data["errors"].append({"stage": stage, "error": error})
        self.log(stage, f"ERROR: {error}")

    def log_improvement(self, benchmark: str, before: float, after: float):
        """Log a benchmark improvement."""
        self.session_data["improvements"][benchmark] = {
            "before": before,
            "after": after,
            "delta": after - before,
        }
        self.log("evaluation", f"{benchmark}: {before:.4f} -> {after:.4f} ({after - before:+.4f})")

    def set_token_usage(self, usage: dict[str, int]):
        """Record token usage for the session."""
        self.session_data["token_usage"] = usage

    def finalize(self, status: str = "completed"):
        """Finalize the session log and write to file."""
        self.session_data["end_time"] = datetime.now(timezone.utc).isoformat()
        self.session_data["status"] = status
        self.log_file.write_text(json.dumps(self.session_data, indent=2))
        print(f"Session log written to {self.log_file}")

    def commit_to_repo(self):
        """Stage all session artifacts and commit to the git repository."""
        try:
            # Stage all evolution artifacts
            paths_to_stage = [
                str(self.log_file),
                "logs/",
                "configs/evolution_state.json",
                "data/generated/",
            ]
            for path in paths_to_stage:
                full = self.repo_root / path
                if full.exists():
                    subprocess.run(
                        ["git", "add", str(full)],
                        cwd=self.repo_root,
                        capture_output=True,
                    )

            # Check if there's anything to commit
            status = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=self.repo_root,
                capture_output=True,
            )
            if status.returncode == 0:
                self.log("commit", "No changes to commit")
                return

            # Build commit message
            msg = f"[terra-evolution] Session {self.session_id}: {self.session_data['status']}"
            stages = ", ".join(self.session_data["stages_completed"])
            if stages:
                msg += f"\n\nStages: {stages}"
            improvements = self.session_data.get("improvements", {})
            if improvements:
                msg += "\n\nImprovements:"
                for bench, vals in improvements.items():
                    msg += f"\n  {bench}: {vals['before']:.4f} -> {vals['after']:.4f} ({vals['delta']:+.4f})"
            errors = self.session_data.get("errors", [])
            if errors:
                msg += f"\n\nErrors: {len(errors)}"
            usage = self.session_data.get("token_usage", {})
            if usage:
                msg += f"\n\nToken usage: {usage}"

            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
            )
            self.log("commit", "Session committed to repository")
        except subprocess.CalledProcessError as e:
            self.log_error("commit", f"Git commit failed: {e.stderr.decode() if e.stderr else str(e)}")

    def get_recent_sessions(self, n: int = 5) -> list[dict]:
        """Load recent session logs for context."""
        sessions = []
        log_files = sorted(self.log_dir.glob("session_*.json"), reverse=True)
        for f in log_files[:n]:
            try:
                sessions.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return sessions
