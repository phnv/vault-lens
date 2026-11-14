"""Minimal provider interface and shared LangChain base. Ollama, Gemini and Groq (docs/adr/0004)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

PROVIDERS = ("off", "ollama", "gemini", "groq")
GROQ_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")
# Providers that send note excerpts off the machine. The UI must say so before first use.
OFF_DEVICE = {"gemini", "groq"}


class LLMError(RuntimeError):
    """Provider unreachable, misconfigured or returned an unusable response (user-safe)."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: int


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> LLMResult: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...

    def health(self) -> tuple[bool, str]: ...


class LangChainChatClient:
    """Shared DRY base for LangChain chat model adapters (Ollama, Gemini, Groq)."""

    provider: str = ""
    model: str = ""
    _chat: Any = None

    @staticmethod
    def _messages(system: str, user: str) -> list[tuple[str, str]]:
        return [("system", system), ("human", user)]

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content or "")

    def _invoke_chat(self, chat: Any, target_model: str, system: str, user: str) -> LLMResult:
        t0 = time.monotonic()
        msg = chat.invoke(self._messages(system, user))
        text = self._extract_text(getattr(msg, "content", ""))
        ms = int((time.monotonic() - t0) * 1000)
        return LLMResult(text, self.provider, target_model, ms)

    def _stream_chat(self, chat: Any, system: str, user: str) -> Iterator[str]:
        for chunk in chat.stream(self._messages(system, user)):
            piece = self._extract_text(getattr(chunk, "content", ""))
            if piece:
                yield piece

    def complete(self, system: str, user: str) -> LLMResult:
        try:
            return self._invoke_chat(self._chat, self.model, system, user)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"{self.provider.title()} request failed ({type(exc).__name__})"
            ) from exc

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            yield from self._stream_chat(self._chat, system, user)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.provider.title()} stream failed ({type(exc).__name__})") from exc
