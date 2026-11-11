from __future__ import annotations

from typing import Any

from langchain_ollama import ChatOllama
from ollama import Client as OllamaSDK

from .base import LangChainChatClient, LLMError, LLMResult


class OllamaClient(LangChainChatClient):
    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        temperature: float = 0.2,
        chat: Any | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.timeout_s, self.temperature = timeout_s, temperature
        self._chat = chat or ChatOllama(
            base_url=self.base_url,
            model=model,
            temperature=temperature,
            client_kwargs={"timeout": timeout_s},
        )
        self._sdk = sdk_client or OllamaSDK(host=self.base_url, timeout=3.0)

    def complete(self, system: str, user: str) -> LLMResult:
        try:
            return self._invoke_chat(self._chat, self.model, system, user)
        except Exception as exc:
            msg = f"Ollama request failed ({type(exc).__name__}). Is the model pulled?"
            raise LLMError(msg) from exc

    def health(self) -> tuple[bool, str]:
        try:
            resp = self._sdk.list()
            raw_models = (
                resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
            )
            names = [
                (m.get("model") or m.get("name", ""))
                if isinstance(m, dict)
                else (getattr(m, "model", None) or getattr(m, "name", ""))
                for m in raw_models
            ]
        except Exception:
            return False, "Ollama not reachable"
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted in names:
            return True, "ready"
        return False, f"model {self.model} not pulled (make pull-model)"
