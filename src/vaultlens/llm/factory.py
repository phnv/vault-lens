from __future__ import annotations

from ..config import LLMConfig
from .base import PROVIDERS, LLMClient, LLMError


def get_client(provider: str, cfg: LLMConfig) -> LLMClient | None:
    """Return a client, or None for provider 'off' (plain ES search, no LLM)."""
    if provider == "off":
        return None
    if provider == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(cfg.ollama_url, cfg.ollama_model, cfg.timeout_s)
    if provider == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(cfg.gemini_api_key, cfg.gemini_model, cfg.gemini_base_url, cfg.timeout_s)
    raise LLMError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
