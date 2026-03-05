"""LLM client layer for external model access (Claude, GLM).

Either or both can be used. At least one API key must be configured:
  - ANTHROPIC_API_KEY for Claude
  - GLM_API_KEY for GLM (z.ai)
"""

import json
import os
from dataclasses import dataclass

import time

import httpx

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    max_retries: int = 3,
    **kwargs,
) -> httpx.Response:
    """Make an HTTP request with exponential backoff on 429/5xx errors."""
    for attempt in range(max_retries + 1):
        resp = client.request(method, url, **kwargs)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                resp.raise_for_status()
            retry_after = int(resp.headers.get("retry-after", 0))
            wait = max(retry_after, 2 ** (attempt + 1))
            print(f"[rate-limit] {resp.status_code} - retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    return resp  # unreachable but satisfies type checker


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict
    raw: dict


class ClaudeClient:
    """Client for Anthropic Claude API. Optional - only used if ANTHROPIC_API_KEY is set."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        if not _HAS_ANTHROPIC:
            raise ImportError("anthropic package required for ClaudeClient. Install with: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.total_tokens_used = 0

    def query(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if temperature != 1.0:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)
        self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw=response.model_dump(),
        )

    def generate_training_data(self, domain: str, num_samples: int = 10) -> list[dict]:
        """Use Claude to generate high-quality training data via distillation."""
        prompt = f"""Generate {num_samples} high-quality instruction-response pairs for the domain: {domain}.

Each pair should:
1. Have a clear, specific instruction
2. Include a detailed, accurate response with reasoning
3. Vary in difficulty from basic to advanced
4. Be self-contained (no external references needed)

Return as a JSON array of objects with "instruction" and "response" fields.
Only return the JSON array, no other text."""

        response = self.query(
            prompt,
            system="You are a expert data generator for training language models. Generate diverse, high-quality training examples.",
            max_tokens=8192,
            temperature=0.8,
        )

        try:
            # Extract JSON from response
            text = response.content
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return []

    def evaluate_response(self, instruction: str, response: str) -> float:
        """Use Claude as a judge to score a response quality (0-1)."""
        prompt = f"""Rate the quality of the following response on a scale of 0 to 10.

Instruction: {instruction}

Response: {response}

Criteria:
- Accuracy: Is the information correct?
- Completeness: Does it fully address the instruction?
- Clarity: Is it well-written and easy to understand?
- Reasoning: Does it show good reasoning when applicable?

Return ONLY a single number (0-10), nothing else."""

        result = self.query(prompt, max_tokens=10, temperature=0.0)
        try:
            score = float(result.content.strip())
            return min(max(score / 10.0, 0.0), 1.0)
        except ValueError:
            return 0.5


class GLMClient:
    """Client for GLM via z.ai API. Can be used as sole LLM or alongside Claude."""

    def __init__(
        self,
        model: str = "glm-4-plus",
        base_url: str | None = None,
    ):
        self.api_key = os.environ["GLM_API_KEY"]
        self.base_url = base_url or os.environ.get(
            "GLM_API_BASE", "https://api.z.ai/v1"
        )
        self.model = model
        self.total_tokens_used = 0
        # Minimum seconds between requests to avoid 429s
        self._min_interval = float(os.environ.get("GLM_REQUEST_INTERVAL", "1.5"))
        self._last_request_time = 0.0

    def query(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Rate-limit: wait between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        with httpx.Client(timeout=120) as client:
            resp = _request_with_retry(
                client,
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            data = resp.json()

        self._last_request_time = time.time()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.total_tokens_used += usage.get("total_tokens", 0)

        return LLMResponse(content=content, model=self.model, usage=usage, raw=data)

    def generate_training_data(self, domain: str, num_samples: int = 10) -> list[dict]:
        """Use GLM to generate training data."""
        prompt = f"""Generate {num_samples} high-quality instruction-response pairs for: {domain}.

Requirements:
1. Clear, specific instructions
2. Detailed, accurate responses with reasoning
3. Varying difficulty levels
4. Self-contained examples

Return as a JSON array with "instruction" and "response" fields. Only JSON, no other text."""

        response = self.query(prompt, max_tokens=8192, temperature=0.8)
        try:
            text = response.content
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return []


class TerraClient:
    """Client for ProjectTerra's own evolved model (self-hosting).

    Once the terra model is good enough, it can be used for research,
    data generation, and evaluation alongside or instead of external LLMs.
    Uses a local inference server (llama.cpp server, MLX server, etc.).
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.model = "terra"
        self.total_tokens_used = 0

    def query(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.total_tokens_used += usage.get("total_tokens", 0)
        return LLMResponse(content=content, model=self.model, usage=usage, raw=data)

    def generate_training_data(self, domain: str, num_samples: int = 10) -> list[dict]:
        """Use terra's own model to generate training data (self-instruct)."""
        prompt = f"""Generate {num_samples} high-quality instruction-response pairs for: {domain}.

Requirements:
1. Clear, specific instructions
2. Detailed, accurate responses with reasoning
3. Varying difficulty levels
4. Self-contained examples

Return as a JSON array with "instruction" and "response" fields. Only JSON, no other text."""

        response = self.query(prompt, max_tokens=8192, temperature=0.8)
        try:
            text = response.content
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return []


class LLMPool:
    """Pool of LLM clients for research and data generation.

    Supports any combination of:
    - Claude (ANTHROPIC_API_KEY) - external
    - GLM (GLM_API_KEY) - external
    - Terra (TERRA_SERVER_URL) - our own evolved model (self-hosting)

    At least one must be configured.
    """

    def __init__(self):
        self.clients: dict[str, ClaudeClient | GLMClient | TerraClient] = {}
        if os.environ.get("ANTHROPIC_API_KEY") and _HAS_ANTHROPIC:
            self.clients["claude"] = ClaudeClient()
        if os.environ.get("GLM_API_KEY"):
            self.clients["glm"] = GLMClient()
        if os.environ.get("TERRA_SERVER_URL"):
            self.clients["terra"] = TerraClient(os.environ["TERRA_SERVER_URL"])
        if not self.clients:
            print(
                "WARNING: No LLM configured. "
                "Set ANTHROPIC_API_KEY, GLM_API_KEY, or TERRA_SERVER_URL."
            )

    def get(self, name: str) -> ClaudeClient | GLMClient | TerraClient | None:
        return self.clients.get(name)

    def get_any(self) -> ClaudeClient | GLMClient | TerraClient | None:
        """Get any available LLM client. Prefers external LLMs, falls back to terra."""
        return self.clients.get("claude") or self.clients.get("glm") or self.clients.get("terra")

    def available(self) -> list[str]:
        return list(self.clients.keys())

    def research(self, question: str) -> dict[str, str]:
        """Query all available LLMs and collect their perspectives."""
        results = {}
        for name, client in self.clients.items():
            try:
                resp = client.query(
                    question,
                    system="You are a research assistant helping improve a small language model. Provide specific, actionable insights.",
                )
                results[name] = resp.content
            except Exception as e:
                results[name] = f"Error: {e}"
        return results

    def generate_data(self, domain: str, samples_per_source: int = 10) -> list[dict]:
        """Generate training data from all available sources."""
        all_data = []
        for name, client in self.clients.items():
            try:
                data = client.generate_training_data(domain, samples_per_source)
                for item in data:
                    item["source"] = name
                all_data.extend(data)
            except Exception as e:
                print(f"[{name}] Data generation failed: {e}")
        return all_data

    def get_token_usage(self) -> dict[str, int]:
        return {name: client.total_tokens_used for name, client in self.clients.items()}
