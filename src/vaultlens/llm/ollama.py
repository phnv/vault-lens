from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx

from .base import LLMError, LLMResult


class OllamaClient:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0, temperature: float = 0.2) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.timeout_s, self.temperature = timeout_s, temperature

    def _payload(self, system: str, user: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self.temperature},
        }

    def complete(self, system: str, user: str) -> LLMResult:
        t0 = time.monotonic()
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=self._payload(system, user, False), timeout=self.timeout_s)
            r.raise_for_status()
            text = r.json()["message"]["content"]
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed ({type(exc).__name__}). Is the model pulled?") from exc
        except (KeyError, ValueError) as exc:
            raise LLMError("Ollama returned an unexpected response") from exc
        return LLMResult(text, self.provider, self.model, int((time.monotonic() - t0) * 1000))

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/api/chat", json=self._payload(system, user, True), timeout=self.timeout_s
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if piece := chunk.get("message", {}).get("content"):
                        yield piece
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama stream failed ({type(exc).__name__})") from exc

    def health(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return False, "Ollama not reachable"
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return (True, "ready") if wanted in names else (False, f"model {self.model} not pulled (make pull-model)")
