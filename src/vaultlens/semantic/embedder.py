"""Embedders (ADR-0009 section 5). Models load lazily on first use, never at import time."""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from ..config import EmbedConfig, Settings
from ..models import Note
from ..search.mapping import mapping_embedding_dims


class EmbeddingError(RuntimeError):
    """User-safe embedding failure (no secrets, no note text)."""


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def chunk_embedding_text(title: str, heading: str, text: str) -> str:
    """'{title} > {heading}\\n{text}'; title and heading omitted when empty."""
    head = " > ".join(p for p in (title.strip(), heading.strip()) if p)
    return f"{head}\n{text}" if head else text


def note_embedding_inputs(note: Note) -> list[str]:
    return [chunk_embedding_text(note.title, c.heading, c.text) for c in note.chunks]


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [float(x) for x in vec] if norm == 0 else [float(x) / norm for x in vec]


def needs_query_prompt(model: str) -> bool:
    """Qwen3-Embedding expects an instruction prefix on queries only; bge-m3 uses none."""
    return "qwen3-embedding" in model.lower()


class SentenceTransformersEmbedder:
    def __init__(self, cfg: EmbedConfig, model_factory=None) -> None:
        self.cfg = cfg
        self._factory = model_factory
        self._model = None

    def _load(self):
        if self._model is None:
            if self._factory is not None:
                self._model = self._factory(self.cfg.model)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.cfg.model)
            self._model.max_seq_length = self.cfg.max_tokens
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._load().encode(list(texts), batch_size=self.cfg.batch, normalize_embeddings=True)
        return [[float(x) for x in v] for v in out]

    def embed_query(self, text: str) -> list[float]:
        kwargs = {"prompt_name": "query"} if needs_query_prompt(self.cfg.model) else {}
        out = self._load().encode([text], normalize_embeddings=True, **kwargs)
        return [float(x) for x in out[0]]


class OllamaEmbedder:
    def __init__(self, cfg: EmbedConfig, base_url: str, timeout_s: float = 120.0) -> None:
        self.cfg, self.base_url, self.timeout_s = cfg, base_url.rstrip("/"), timeout_s

    def _embed(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps({"model": self.cfg.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                data = json.loads(resp.read())
        except Exception as exc:
            raise EmbeddingError(f"Ollama embedding failed ({type(exc).__name__})") from exc
        return [l2_normalize(v) for v in data["embeddings"]]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        items = list(texts)
        for i in range(0, len(items), self.cfg.batch):
            out += self._embed(items[i : i + self.cfg.batch])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def validate_dims(cfg: EmbedConfig) -> None:
    mapped = mapping_embedding_dims()
    if mapped is not None and cfg.dims != mapped:
        raise EmbeddingError(f"EMBED_DIMS={cfg.dims} does not match the index mapping ({mapped})")


@lru_cache(maxsize=4)
def _cached(cfg: EmbedConfig, ollama_url: str) -> Embedder:
    validate_dims(cfg)
    if cfg.provider == "ollama":
        return OllamaEmbedder(cfg, ollama_url)
    return SentenceTransformersEmbedder(cfg)


def get_embedder(settings: Settings) -> Embedder:
    """Cached singleton per config; the model itself loads on first encode."""
    return _cached(settings.embed, settings.llm.ollama_url)
