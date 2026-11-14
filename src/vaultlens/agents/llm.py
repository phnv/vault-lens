"""Chat model factory for AI Search and Agentic tiers (ADR-0009 section 6.8 / R2).

Supports `groq` and `gemini` only via `langchain-groq` and `langchain-google-genai`.
API keys come from `LLMConfig` only and never appear in logs or URLs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from ..config import LLMConfig
from ..llm.base import LLMError

AGENT_PROVIDERS = ("groq", "gemini")


def get_chat_model(
    provider: str,
    cfg: LLMConfig,
    *,
    groq_model: str | None = None,
    temperature: float = 0.2,
) -> BaseChatModel:
    if provider == "off":
        raise LLMError("LLM provider is 'off'; select Groq or Gemini.")
    if provider == "groq":
        if not cfg.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set")
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=cfg.groq_api_key,  # type: ignore[arg-type]
            model=groq_model or cfg.groq_model,
            temperature=temperature,
            timeout=cfg.timeout_s,
            max_retries=0,
        )
    if provider == "gemini":
        if not cfg.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=cfg.gemini_api_key,
            model=cfg.gemini_model,
            temperature=temperature,
            timeout=cfg.timeout_s,
            max_retries=0,
        )
    raise LLMError(
        f"Provider {provider!r} is not supported for AI Search / Agentic tiers; "
        f"choose one of {AGENT_PROVIDERS}."
    )


def invoke_structured_with_repair(
    model: Any,
    schema: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
) -> tuple[Any, int]:
    """Call `model.with_structured_output(schema)`, retrying once on validation/parse failure.

    Returns `(parsed_instance, llm_calls_used)`. Raises `LLMError` if the repair retry also fails.
    """
    structured = model.with_structured_output(schema)
    messages = [("system", system_prompt), ("human", user_prompt)]
    try:
        out = structured.invoke(messages)
        if isinstance(out, dict):
            out = schema.model_validate(out)
        elif not isinstance(out, schema):
            out = schema.model_validate(out)
        return out, 1
    except Exception as first_exc:
        repair_user = (
            f"{user_prompt}\n\n"
            f"Your previous output failed schema validation ({type(first_exc).__name__}: "
            f"{str(first_exc)[:240]}). Return valid structured output matching `{schema.__name__}`."
        )
        try:
            out2 = structured.invoke([("system", system_prompt), ("human", repair_user)])
            if isinstance(out2, dict):
                out2 = schema.model_validate(out2)
            elif not isinstance(out2, schema):
                out2 = schema.model_validate(out2)
            return out2, 2
        except Exception as second_exc:
            raise LLMError(
                f"Structured output for {schema.__name__} failed after repair retry "
                f"({type(second_exc).__name__})"
            ) from second_exc
