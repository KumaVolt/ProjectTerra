"""Triage script for GitHub Actions - determines what to work on this session.

Works with either Claude (ANTHROPIC_API_KEY) or GLM (GLM_API_KEY) or both.
"""

import json
import os
from pathlib import Path

import httpx


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Call whichever LLM is available (Claude or GLM)."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    glm_key = os.environ.get("GLM_API_KEY")

    if anthropic_key:
        return _call_claude(prompt, system, max_tokens, anthropic_key)
    elif glm_key:
        return _call_glm(prompt, system, max_tokens, glm_key)
    else:
        raise RuntimeError("No LLM API key configured. Set ANTHROPIC_API_KEY or GLM_API_KEY.")


def _call_claude(prompt: str, system: str, max_tokens: int, api_key: str) -> str:
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    usage = data.get("usage", {})
    print(f"[triage] LLM=claude tokens_in={usage.get('input_tokens', 0)} tokens_out={usage.get('output_tokens', 0)}")
    return data["content"][0]["text"]


def _call_glm(prompt: str, system: str, max_tokens: int, api_key: str) -> str:
    base_url = os.environ.get("GLM_API_BASE", "https://api.z.ai/api/paas/v4")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Try multiple known GLM API endpoints
    urls_to_try = [
        base_url,
        "https://api.z.ai/api/paas/v4",
        "https://api.z.ai/v1",
    ]
    # Deduplicate while preserving order
    seen = set()
    urls_to_try = [u for u in urls_to_try if not (u in seen or seen.add(u))]

    last_error = None
    for url in urls_to_try:
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{url}/chat/completions",
                    json={"model": "glm-4.5-air", "max_tokens": max_tokens, "messages": messages},
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            usage = data.get("usage", {})
            print(f"[triage] LLM=glm url={url} tokens={usage.get('total_tokens', 0)}")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            print(f"[triage] GLM endpoint {url} failed: {e}")
            continue

    raise RuntimeError(f"All GLM endpoints failed. Last error: {last_error}")


def load_context():
    """Load context gathered by the workflow."""
    issues = json.loads(Path("/tmp/issues.json").read_text()) if Path("/tmp/issues.json").exists() else []
    recent_commits = Path("/tmp/recent_commits.txt").read_text() if Path("/tmp/recent_commits.txt").exists() else ""
    recent_sessions = []
    try:
        raw = Path("/tmp/recent_sessions.json").read_text()
        if raw.strip():
            recent_sessions = json.loads(raw) if raw.strip().startswith("[") else []
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return issues, recent_commits, recent_sessions


def generate_work_plan(issues, recent_commits, recent_sessions):
    """Use an LLM to generate a work plan for this session."""
    cycle_mode = os.environ.get("CYCLE_MODE", "full")
    target_issue = os.environ.get("TARGET_ISSUE", "")

    context = f"""## Open Issues
{json.dumps(issues[:20], indent=2)}

## Recent Commits
{recent_commits}

## Recent Session Results
{json.dumps(recent_sessions[:5], indent=2)}

## Cycle Mode: {cycle_mode}
## Target Issue: {target_issue or 'None'}"""

    prompt = f"""You are the planning module for ProjectTerra, a self-evolving multimodal LLM.
Your job is to decide what this evolution cycle should focus on.

{context}

Based on the above context, create a work plan. Return a JSON object with:
{{
  "phases": ["research", "data_generation", "training", "evaluation"],
  "focus_areas": ["list of domains to focus on"],
  "issues_to_address": [list of issue numbers],
  "data_domains": ["reasoning", "coding", etc.],
  "strategy_notes": "brief explanation of why this plan",
  "estimated_impact": "low/medium/high"
}}

If mode is 'research-only', only include research phase.
If mode is 'evaluate-only', only include evaluation phase.
If mode is 'fix-issue', focus on the target issue.
Otherwise, include all phases that make sense given context."""

    text = call_llm(prompt)

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])

    return {
        "phases": ["research", "data_generation", "training", "evaluation"],
        "focus_areas": ["reasoning", "coding", "math"],
        "issues_to_address": [],
        "data_domains": ["reasoning", "coding", "math"],
        "strategy_notes": "Default plan - could not parse LLM response",
        "estimated_impact": "medium",
    }


def main():
    issues, recent_commits, recent_sessions = load_context()

    has_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GLM_API_KEY")
    if has_api_key:
        plan = generate_work_plan(issues, recent_commits, recent_sessions)
    else:
        plan = {
            "phases": ["research", "data_generation", "training", "evaluation"],
            "focus_areas": ["reasoning", "coding", "math"],
            "issues_to_address": [i["number"] for i in issues[:3]] if issues else [],
            "data_domains": ["reasoning", "coding", "math"],
            "strategy_notes": "Default plan (no API key)",
            "estimated_impact": "medium",
        }

    plan_json = json.dumps(plan)
    print(f"Work plan: {plan_json}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"work_plan={plan_json}\n")
    else:
        print(f"::set-output name=work_plan::{plan_json}")


if __name__ == "__main__":
    main()
