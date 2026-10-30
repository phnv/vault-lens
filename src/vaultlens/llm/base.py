"""Minimal provider interface. Only Ollama and Gemini are supported (docs/adr/0004)."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

PROVIDERS = ("off", "ollama", "gemini")
# Providers that send note excerpts off the machine. The UI must say so before first use.
OFF_DEVICE = {"gemini"}


class LLMError(RuntimeError):
    """Provider unreachable, misconfigured or returned an unusable response. Message is user-safe."""


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
