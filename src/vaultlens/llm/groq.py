from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from groq import APIError, APIStatusError, Groq, RateLimitError
from langchain_groq import ChatGroq

from .base import GROQ_MODELS, LangChainChatClient, LLMError, LLMResult

DEFAULT_WAIT_S = 60.0
MAX_WAIT_S = 65.0

_DURATION_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?"
)


def parse_duration_s(raw: str | None) -> float | None:
    """Parse Groq header durations ('7.66s', '2m59.56s', '2') into seconds."""
    if not raw or not (s := raw.strip()):
        return None
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    m = _DURATION_RE.fullmatch(s)
    if not m or not any(m.groups()):
        return None
    h, m_, sec, ms = (float(g or 0.0) for g in m.groups())
    return max(0.0, h * 3600.0 + m_ * 60.0 + sec + ms / 1000.0)


def _headers_of(exc: RateLimitError) -> Mapping[str, str]:
    resp = getattr(exc, "response", None)
    return getattr(resp, "headers", {}) or {}


def _is_daily_exhausted(exc: RateLimitError) -> bool:
    """True when a RateLimitError indicates the daily (RPD/TPD) quota is exhausted."""
    headers = _headers_of(exc)
    rem_req = headers.get("x-ratelimit-remaining-requests", "").strip()
    if rem_req.isdigit() and int(rem_req) <= 0:
        return True
    msg = str(exc).lower()
    return "per day" in msg or "(tpd)" in msg or "(rpd)" in msg


def _wait_seconds(exc: RateLimitError, default: float = DEFAULT_WAIT_S) -> float:
    """Extract wait time from retry-after or x-ratelimit-reset-tokens on 429."""
    headers = _headers_of(exc)
    for key in ("retry-after", "x-ratelimit-reset-tokens"):
        val = parse_duration_s(headers.get(key))
        if val is not None and val > 0:
            return min(val, MAX_WAIT_S)
    return default


class GroqClient(LangChainChatClient):
    provider = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = GROQ_MODELS[0],
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        chat_factory: Callable[[str], Any] | None = None,
        sdk_client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self._key = api_key  # never log or display in URLs or the UI
        self.model = model
        self.timeout_s, self.temperature = timeout_s, temperature
        self._sleep = sleep_fn
        self._chat_factory = chat_factory or self._build_chat
        self._sdk = sdk_client

    def _build_chat(self, target_model: str) -> Any:
        kwargs: dict[str, Any] = {
            "groq_api_key": self._key,
            "model_name": target_model,
            "temperature": self.temperature,
            "request_timeout": self.timeout_s,
            "max_retries": 0,
        }
        if target_model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"
        return ChatGroq(**kwargs)

    def _models(self) -> list[str]:
        return [self.model, *[m for m in GROQ_MODELS if m != self.model]]

    def complete(self, system: str, user: str) -> LLMResult:
        errors_429: list[RateLimitError] = []
        try:
            for target in self._models():
                try:
                    return self._invoke_chat(self._chat_factory(target), target, system, user)
                except RateLimitError as exc:
                    errors_429.append(exc)
                    continue

            # Both models returned 429: inspect rate-limit headers now
            minute_waits = [_wait_seconds(e) for e in errors_429 if not _is_daily_exhausted(e)]
            if not minute_waits:
                raise LLMError("Groq daily quota (RPD/TPD) exhausted across available models.")
            self._sleep(min(minute_waits))
            try:
                return self._invoke_chat(self._chat_factory(self.model), self.model, system, user)
            except RateLimitError as exc:
                raise LLMError("Groq rate limit (429) persisted after fallback and retry.") from exc
        except LLMError:
            raise
        except APIStatusError as exc:
            raise LLMError(f"Groq returned HTTP {exc.status_code}") from exc
        except APIError as exc:
            raise LLMError(f"Groq request failed ({type(exc).__name__})") from exc
        except Exception as exc:
            raise LLMError(f"Groq request failed ({type(exc).__name__})") from exc

    def stream(self, system: str, user: str) -> Iterator[str]:
        errors_429: list[RateLimitError] = []
        try:
            for target in self._models():
                try:
                    yield from self._stream_chat(self._chat_factory(target), system, user)
                    return
                except RateLimitError as exc:
                    errors_429.append(exc)
                    continue

            # Both models returned 429: inspect rate-limit headers now
            minute_waits = [_wait_seconds(e) for e in errors_429 if not _is_daily_exhausted(e)]
            if not minute_waits:
                raise LLMError("Groq daily quota (RPD/TPD) exhausted across available models.")
            self._sleep(min(minute_waits))
            try:
                yield from self._stream_chat(self._chat_factory(self.model), system, user)
            except RateLimitError as exc:
                raise LLMError("Groq rate limit (429) persisted after fallback and retry.") from exc
        except LLMError:
            raise
        except APIStatusError as exc:
            raise LLMError(f"Groq returned HTTP {exc.status_code}") from exc
        except APIError as exc:
            raise LLMError(f"Groq stream failed ({type(exc).__name__})") from exc
        except Exception as exc:
            raise LLMError(f"Groq stream failed ({type(exc).__name__})") from exc

    def health(self) -> tuple[bool, str]:
        try:
            sdk = self._sdk or Groq(api_key=self._key, timeout=5.0, max_retries=0)
            sdk.models.list()
            return True, "ready"
        except APIStatusError as exc:
            return False, f"Groq HTTP {exc.status_code}"
        except Exception:
            return False, "Groq not reachable"
