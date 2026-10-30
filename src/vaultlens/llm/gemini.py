from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx

from .base import LLMError, LLMResult


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout_s: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self._key = api_key  # sent as a header only: never in URLs, logs or the UI
        self.model, self.base_url = model, base_url.rstrip("/")
        self.timeout_s, self.temperature = timeout_s, temperature

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._key, "content-type": "application/json"}

    def _payload(self, system: str, user: str) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature},
        }

    @staticmethod
    def _text(data: dict) -> str:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    def complete(self, system: str, user: str) -> LLMResult:
        t0 = time.monotonic()
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        try:
            r = httpx.post(url, headers=self._headers(), json=self._payload(system, user), timeout=self.timeout_s)
            r.raise_for_status()
            text = self._text(r.json())
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Gemini returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed ({type(exc).__name__})") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Gemini returned an unexpected or empty response") from exc
        return LLMResult(text, self.provider, self.model, int((time.monotonic() - t0) * 1000))

    def stream(self, system: str, user: str) -> Iterator[str]:
        url = f"{self.base_url}/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        try:
            with httpx.stream(
                "POST", url, headers=self._headers(), json=self._payload(system, user), timeout=self.timeout_s
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        try:
                            piece = self._text(json.loads(line[5:].strip()))
                        except (KeyError, IndexError, ValueError):
                            continue
                        if piece:
                            yield piece
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Gemini returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini stream failed ({type(exc).__name__})") from exc

    def health(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/v1beta/models/{self.model}", headers=self._headers(), timeout=5)
        except httpx.HTTPError:
            return False, "Gemini not reachable"
        return (True, "ready") if r.status_code == 200 else (False, f"Gemini HTTP {r.status_code}")
