from __future__ import annotations

from typing import Any

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import LangChainChatClient, LLMError


class GeminiClient(LangChainChatClient):
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        chat: Any | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self._key = api_key  # never log or display in URLs or the UI
        self.model = model
        self.timeout_s, self.temperature = timeout_s, temperature
        self._chat = chat or ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout_s,
            max_retries=0,
        )
        self._sdk = sdk_client

    def health(self) -> tuple[bool, str]:
        try:
            sdk = self._sdk or genai.Client(api_key=self._key, http_options={"timeout": 5000})
            sdk.models.get(model=self.model)
            return True, "ready"
        except Exception as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            return (False, f"Gemini HTTP {code}") if code else (False, "Gemini not reachable")
