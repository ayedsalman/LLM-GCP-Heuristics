"""Minimal LLM clients (plan B1, B7). Standard library only (urllib).

Every call returns the provider's *reported* model snapshot, the query time,
and the raw JSON, which go into the candidate registry (S0.3). As in the
paper: no system prompt, temperature 0.7 unless stated.

Model table: registry/models.json. Requested model IDs in that file must be
checked by the authors before running: if a snapshot used in the paper has
been retired, use the closest available one and report the change (B7).
"""

from __future__ import annotations

import datetime
import json
import os
import random
import time
import zlib
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model_requested: str
    model_snapshot: str
    queried_at_utc: str
    temperature: float
    seed: int | None
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _post(url: str, headers: dict, body: dict, timeout: int = 300, retries: int = 4) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(2 ** attempt * 5)
                continue
            raise RuntimeError(f"{url}: HTTP {e.code}: {e.read().decode()[:500]}") from e
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(2 ** attempt * 5)
                continue
            raise


class OpenAICompatibleClient:
    """OpenAI, DeepSeek, Together, Groq, OpenRouter, vLLM ... (/v1/chat/completions)."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", send_seed: bool = True,
                 max_tokens: int | None = None, token_param: str = "max_tokens"):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "")
        self.key_env = api_key_env
        self.send_seed = send_seed
        self.max_tokens, self.token_param = max_tokens, token_param

    def complete(self, messages, temperature: float = 0.7, seed: int | None = None) -> LLMResponse:
        if not self.key:
            raise RuntimeError(f"set {self.key_env}")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.send_seed and seed is not None:
            body["seed"] = seed
        if self.max_tokens:
            body[self.token_param] = self.max_tokens
        raw = _post(f"{self.base_url}/chat/completions", {"authorization": f"Bearer {self.key}"}, body)
        text = raw["choices"][0]["message"]["content"] or ""
        return LLMResponse(text, self.model, raw.get("model", self.model), _now(), temperature, seed,
                           raw.get("usage", {}), raw)


class AnthropicClient:
    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY", max_tokens: int = 4096,
                 base_url: str = "https://api.anthropic.com/v1"):
        self.model, self.max_tokens, self.base_url = model, max_tokens, base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "")
        self.key_env = api_key_env

    def complete(self, messages, temperature: float = 0.7, seed: int | None = None) -> LLMResponse:
        if not self.key:
            raise RuntimeError(f"set {self.key_env}")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages,
                "temperature": temperature}
        raw = _post(f"{self.base_url}/messages",
                    {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        return LLMResponse(text, self.model, raw.get("model", self.model), _now(), temperature, seed,
                           raw.get("usage", {}), raw)


class MockClient:
    """Deterministic stand-in for tests and dry runs. Writes responses in the
    format the universal prompt asks for, with saturation-led random weights,
    and occasionally a bilinear term (to exercise the validator)."""

    def __init__(self, model: str = "mock", p_violation: float = 0.15, cuda_ok: float = 0.5):
        self.model = model
        self.p_violation = p_violation
        self.cuda_ok = cuda_ok
        self.calls = 0

    def complete(self, messages, temperature: float = 0.7, seed: int | None = None) -> LLMResponse:
        self.calls += 1
        prompt = messages if isinstance(messages, str) else messages[-1]["content"]
        rng = random.Random(zlib.crc32(repr((self.model, seed, self.calls, prompt)).encode()))
        if ("CUDA" in prompt and "kernel" in prompt) or "program you wrote failed" in prompt:
            ok = rng.random() < self.cuda_ok
            code = "int main(){return 0;}" if ok else "int main( {"
            text = f"H1: score = 1e6*saturation(v) + degree(v)\n```cuda\n{code}\n```"
        else:
            k = 2 if "two refined variants" in prompt or "variants, A and B" in prompt else 5
            if "Propose" in prompt and "new heuristics" in prompt:
                k = int(prompt.split("Propose ")[1].split()[0])
            parts = []
            for i in range(k):
                w = [round(rng.uniform(0.5, 3), 2), round(rng.uniform(-0.5, 0.5), 2),
                     round(rng.uniform(0, 0.5), 2), round(rng.uniform(-1, 1), 2), round(rng.uniform(0, 0.05), 3)]
                f = (f"{w[0]}·saturation(v) + {w[1]}·degree(v) + {w[2]}·uncolored_neighbors(v) "
                     f"+ {w[3]}·clustering_coefficient(v) + {w[4]}·two_hop_neighbors(v)")
                if rng.random() < self.p_violation:
                    f += " + 0.1·degree(v)·clustering_coefficient(v)"
                label = f"Variant {'AB'[i]}" if k == 2 else f"Heuristic {i + 1}"
                parts.append(f"### {label}: Mock-{i}\n**Formula:** score(v) = {f}\n"
                             f"**Design intuition:** saturation first, small corrections.\n"
                             f"**Tie-breaking:** higher degree(v), then lower vertex id.\n")
            text = "\n".join(parts)
        return LLMResponse(text, self.model, f"{self.model}-snapshot", _now(), temperature, seed, {}, {})


def load_models(path: str = "registry/models.json") -> dict:
    with open(path) as f:
        return json.load(f)


def make_client(name: str, models: dict | None = None):
    models = models or load_models()
    cfg = dict(models[name])
    prov = cfg.pop("provider")
    cfg.pop("note", None)
    if prov == "anthropic":
        return AnthropicClient(**cfg)
    if prov == "mock":
        return MockClient(model=name)
    return OpenAICompatibleClient(**cfg)
