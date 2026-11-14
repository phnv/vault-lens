from __future__ import annotations

from types import SimpleNamespace

import pytest

from vaultlens.llm.base import LLMError
from vaultlens.llm.gemini import GeminiClient
from vaultlens.llm.ollama import OllamaClient


class FakeChat:
    def __init__(self, reply="Answer [1]", chunks=("Part 1 ", "Part 2")):
        self.reply = reply
        self.chunks = chunks
        self.invocations: list = []

    def invoke(self, messages):
        self.invocations.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(content=self.reply)

    def stream(self, messages):
        self.invocations.append(messages)
        if isinstance(self.chunks, Exception):
            raise self.chunks
        for c in self.chunks:
            yield SimpleNamespace(content=c)


def test_ollama_complete_and_stream_via_shared_langchain_base():
    fake_chat = FakeChat(reply=[{"text": "Block 1 "}, {"text": "Block 2"}])
    fake_sdk = SimpleNamespace(list=lambda: {"models": [{"model": "qwen2.5:1.5b-instruct"}]})
    client = OllamaClient(
        "http://localhost:11434", "qwen2.5:1.5b-instruct", chat=fake_chat, sdk_client=fake_sdk
    )

    res = client.complete("sys", "question")
    assert res.text == "Block 1 Block 2" and res.provider == "ollama"
    assert list(client.stream("sys", "question")) == ["Part 1 ", "Part 2"]
    assert client.health() == (True, "ready")


def test_ollama_health_when_model_not_pulled_or_down():
    sdk_empty = SimpleNamespace(list=lambda: {"models": [{"name": "other:latest"}]})
    c1 = OllamaClient(
        "http://localhost:11434", "qwen2.5:1.5b-instruct", chat=FakeChat(), sdk_client=sdk_empty
    )
    ok, msg = c1.health()
    assert not ok and "not pulled" in msg

    def _boom():
        raise ConnectionError("down")

    sdk_down = SimpleNamespace(list=_boom)
    c2 = OllamaClient(
        "http://localhost:11434", "qwen2.5:1.5b-instruct", chat=FakeChat(), sdk_client=sdk_down
    )
    assert c2.health() == (False, "Ollama not reachable")


def test_gemini_requires_key_and_uses_shared_langchain_base():
    with pytest.raises(LLMError, match="GEMINI_API_KEY is not set"):
        GeminiClient("", "gemini-2.5-flash")

    fake_chat = FakeChat(reply="Gemini answer [1]")
    fake_models = SimpleNamespace(get=lambda model: SimpleNamespace(name=model))
    fake_sdk = SimpleNamespace(models=fake_models)
    client = GeminiClient("gem-key", "gemini-2.5-flash", chat=fake_chat, sdk_client=fake_sdk)

    res = client.complete("sys", "user")
    assert res.text == "Gemini answer [1]" and res.provider == "gemini"
    assert list(client.stream("sys", "user")) == ["Part 1 ", "Part 2"]
    assert client.health() == (True, "ready")
