"""Main entry point for evolution sessions (used by GitHub Actions)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.orchestrator import EvolutionOrchestrator


def main():
    work_plan = os.environ.get("WORK_PLAN")
    config_path = os.environ.get("CONFIG_PATH", "configs/terra.yaml")

    orchestrator = EvolutionOrchestrator(config_path)

    if work_plan:
        try:
            plan = json.loads(work_plan)
            print(f"Using work plan: {json.dumps(plan, indent=2)}")
        except json.JSONDecodeError:
            print("Could not parse work plan, using default")

    orchestrator.run()


if __name__ == "__main__":
    main()
