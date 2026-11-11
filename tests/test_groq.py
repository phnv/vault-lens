from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from groq import RateLimitError

from vaultlens.llm.base import GROQ_MODELS, LLMError
from vaultlens.llm.groq import GroqClient, parse_duration_s


def _make_429(
    headers: dict[str, str] | None = None,
    message: str = "Rate limit reached (TPM)",
) -> RateLimitError:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, headers=headers or {}, request=req)
    return RateLimitError(message, response=resp, body={"error": {"message": message}})


class FakeChatFactory:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[tuple[str, list]] = []

    def __call__(self, model: str):
        parent = self

        class _FakeChat:
            def invoke(self, messages):
                parent.calls.append((model, messages))
                item = parent._responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return SimpleNamespace(content=item)

            def stream(self, messages):
                parent.calls.append((model, messages))
                item = parent._responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                for piece in item:
                    yield SimpleNamespace(content=piece)

        return _FakeChat()


def test_missing_api_key_raises():
    with pytest.raises(LLMError, match="GROQ_API_KEY is not set"):
        GroqClient("")


def test_parse_duration_strings():
    assert parse_duration_s("2") == 2.0
    assert parse_duration_s("7.66s") == pytest.approx(7.66)
    assert parse_duration_s("2m59.56s") == pytest.approx(179.56)
    assert parse_duration_s(None) is None
    assert parse_duration_s("invalid") is None


def test_complete_happy_path_uses_langchain_chat_groq():
    factory = FakeChatFactory(["Answer [1]"])
    sleeps: list[float] = []
    client = GroqClient(
        "gsk_test",
        model="openai/gpt-oss-120b",
        chat_factory=factory,
        sleep_fn=sleeps.append,
    )

    res = client.complete("sys prompt", "user prompt")
    assert res.text == "Answer [1]"
    assert res.provider == "groq"
    assert res.model == "openai/gpt-oss-120b"
    assert sleeps == []
    expected_msgs = [("system", "sys prompt"), ("human", "user prompt")]
    assert factory.calls == [("openai/gpt-oss-120b", expected_msgs)]


def test_complete_falls_back_to_second_model_immediately_on_single_429():
    factory = FakeChatFactory([
        _make_429({"retry-after": "15", "x-ratelimit-remaining-requests": "900"}),
        "Fallback answer [2]",
    ])
    sleeps: list[float] = []
    client = GroqClient(
        "gsk_test", model=GROQ_MODELS[0], chat_factory=factory, sleep_fn=sleeps.append
    )

    res = client.complete("sys", "user")
    assert res.text == "Fallback answer [2]"
    assert res.model == GROQ_MODELS[1]
    assert sleeps == []  # no sleep when the fallback model succeeds
    assert [m for m, _ in factory.calls] == [GROQ_MODELS[0], GROQ_MODELS[1]]


def test_complete_parses_headers_and_sleeps_only_when_both_models_return_429():
    factory = FakeChatFactory([
        _make_429({"retry-after": "12", "x-ratelimit-remaining-requests": "500"}),
        _make_429({"x-ratelimit-reset-tokens": "7.66s", "x-ratelimit-remaining-requests": "800"}),
        "Recovered after sleep [1]",
    ])
    sleeps: list[float] = []
    client = GroqClient(
        "gsk_test", model=GROQ_MODELS[0], chat_factory=factory, sleep_fn=sleeps.append
    )

    res = client.complete("sys", "user")
    assert res.text == "Recovered after sleep [1]"
    assert res.model == GROQ_MODELS[0]
    assert sleeps == [pytest.approx(7.66)]
    assert [m for m, _ in factory.calls] == [GROQ_MODELS[0], GROQ_MODELS[1], GROQ_MODELS[0]]


def test_complete_fails_fast_without_sleep_when_both_models_hit_daily_rpd_limit():
    rpd_msg = "Rate limit reached on requests per day (RPD)"
    factory = FakeChatFactory([
        _make_429({"x-ratelimit-remaining-requests": "0"}, rpd_msg),
        _make_429({"x-ratelimit-remaining-requests": "0"}, rpd_msg),
    ])
    sleeps: list[float] = []
    client = GroqClient(
        "gsk_test", model=GROQ_MODELS[0], chat_factory=factory, sleep_fn=sleeps.append
    )

    with pytest.raises(LLMError, match="daily quota"):
        client.complete("sys", "user")
    assert sleeps == []


def test_stream_falls_back_and_sleeps_on_both_429():
    factory = FakeChatFactory([
        _make_429({"retry-after": "5"}),
        _make_429({"retry-after": "3"}),
        ["Hello ", "", "world"],
    ])
    sleeps: list[float] = []
    client = GroqClient(
        "gsk_test", model=GROQ_MODELS[0], chat_factory=factory, sleep_fn=sleeps.append
    )

    chunks = list(client.stream("sys", "user"))
    assert "".join(chunks) == "Hello world"
    assert sleeps == [3.0]
