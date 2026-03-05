"""Research agent that uses external LLMs to discover improvements."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.core.llm_client import LLMPool


@dataclass
class ResearchFinding:
    topic: str
    source: str
    summary: str
    actionable_items: list[str] = field(default_factory=list)
    priority: int = 5  # 1=highest, 10=lowest


class ResearchAgent:
    """Autonomous research agent that discovers improvement opportunities."""

    def __init__(self, llm_pool: LLMPool, config: dict):
        self.llm_pool = llm_pool
        self.config = config
        self.findings_dir = Path("logs/research")
        self.findings_dir.mkdir(parents=True, exist_ok=True)

    def research_model_improvements(self) -> list[ResearchFinding]:
        """Research ways to improve the current model."""
        model_name = self.config["model"]["base"]
        findings = []

        questions = [
            {
                "topic": "architecture",
                "question": f"What architectural modifications could improve {model_name} for consumer hardware deployment? Consider attention mechanisms, layer configurations, and novel approaches. Focus on changes that maintain or reduce model size while improving quality.",
            },
            {
                "topic": "training_techniques",
                "question": "What are the most effective training techniques for small language models in 2025-2026? Include specific methods like curriculum learning, knowledge distillation, data mixing strategies, and novel loss functions.",
            },
            {
                "topic": "data_quality",
                "question": "What datasets and data curation strategies have proven most effective for training small but capable language models? Include specific dataset names, mixing ratios, and quality filtering approaches.",
            },
            {
                "topic": "quantization",
                "question": "What are the latest advances in model quantization and compression for mobile/edge deployment? Compare GGUF quantization levels, AWQ, GPTQ, and any newer methods. What gives the best quality-to-size ratio?",
            },
            {
                "topic": "multimodal",
                "question": "How can a small (3B parameter) model best support multimodal capabilities: vision, speech-to-text, text-to-speech, and full-duplex communication? What adapter or modular approaches work at this scale?",
            },
            {
                "topic": "benchmarks",
                "question": "What specific techniques can boost scores on MMLU, ARC, HellaSwag, GSM8K, and HumanEval for models under 4GB? What training data and methods are most impactful for each benchmark?",
            },
        ]

        for q in questions:
            responses = self.llm_pool.research(q["question"])
            for source, response in responses.items():
                finding = self._parse_finding(q["topic"], source, response)
                if finding:
                    findings.append(finding)

        self._save_findings(findings)
        return findings

    def research_specific_topic(self, topic: str, question: str) -> list[ResearchFinding]:
        """Research a specific topic based on an issue or identified need."""
        responses = self.llm_pool.research(question)
        findings = []
        for source, response in responses.items():
            finding = self._parse_finding(topic, source, response)
            if finding:
                findings.append(finding)
        return findings

    def synthesize_findings(self, findings: list[ResearchFinding]) -> dict:
        """Use an LLM to synthesize research findings into an action plan."""
        client = self.llm_pool.get_any()
        if not client:
            return {"actions": [f.actionable_items for f in findings]}

        findings_text = "\n\n".join(
            f"## {f.topic} (source: {f.source})\n{f.summary}\nActions: {', '.join(f.actionable_items)}"
            for f in findings
        )

        prompt = f"""Synthesize these research findings into a prioritized action plan for improving a small LLM.

{findings_text}

Return a JSON object with:
- "immediate_actions": list of things to do this session
- "short_term": list of things to do in next 5 sessions
- "long_term": list of strategic improvements
- "data_priorities": domains to focus data generation on
- "architecture_changes": any model architecture changes to consider

Prioritize by expected impact. Be specific and actionable."""

        response = client.query(prompt, max_tokens=4096, temperature=0.3)
        try:
            text = response.content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        return {"immediate_actions": [item for f in findings for item in f.actionable_items[:2]]}

    def _parse_finding(self, topic: str, source: str, response: str) -> ResearchFinding | None:
        """Parse an LLM response into a structured finding."""
        if response.startswith("Error:"):
            return None

        # Try to extract actionable items
        actions = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
                clean = line.lstrip("-*0123456789. ")
                if len(clean) > 20 and any(
                    kw in clean.lower()
                    for kw in ["use", "implement", "try", "apply", "train", "add", "optimize", "consider"]
                ):
                    actions.append(clean)

        return ResearchFinding(
            topic=topic,
            source=source,
            summary=response[:1000],
            actionable_items=actions[:5],
        )

    def _save_findings(self, findings: list[ResearchFinding]):
        """Save research findings to disk."""
        data = [
            {
                "topic": f.topic,
                "source": f.source,
                "summary": f.summary,
                "actionable_items": f.actionable_items,
                "priority": f.priority,
            }
            for f in findings
        ]
        from datetime import datetime, timezone
        path = self.findings_dir / f"research_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(data, indent=2))
