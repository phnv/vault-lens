"""LangChain-backed embedders (ADR-0009 section 5). Models load lazily on first use."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Protocol

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


QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)


class LangChainEmbedder:
    """Lazy-loading adapter around any `langchain_core.embeddings.Embeddings` implementation."""

    def __init__(
        self,
        cfg: EmbedConfig,
        factory: Callable[[], Any],
        *,
        batch_documents: bool = False,
        query_prefix: str = "",
    ) -> None:
        self.cfg = cfg
        self._factory = factory
        self._embeddings: Any = None
        self._batch_documents = batch_documents
        self._query_prefix = query_prefix

    def _load(self) -> Any:
        if self._embeddings is None:
            self._embeddings = self._factory()
        return self._embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        items = list(texts)
        try:
            emb = self._load()
            if self._batch_documents:
                raw: list[list[float]] = []
                for i in range(0, len(items), self.cfg.batch):
                    raw += emb.embed_documents(items[i : i + self.cfg.batch])
            else:
                raw = emb.embed_documents(items)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"{self.cfg.provider} embedding failed ({type(exc).__name__})"
            ) from exc
        return [l2_normalize(v) for v in raw]

    def embed_query(self, text: str) -> list[float]:
        q = f"{self._query_prefix}{text}" if self._query_prefix else text
        try:
            vec = self._load().embed_query(q)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"{self.cfg.provider} embedding failed ({type(exc).__name__})"
            ) from exc
        return l2_normalize(vec)


def _build_huggingface_local(cfg: EmbedConfig) -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    query_kw: dict[str, Any] = {"normalize_embeddings": True}
    if needs_query_prompt(cfg.model):
        query_kw["prompt_name"] = "query"
    emb = HuggingFaceEmbeddings(
        model_name=cfg.model,
        encode_kwargs={"batch_size": cfg.batch, "normalize_embeddings": True},
        query_encode_kwargs=query_kw,
    )
    emb._client.max_seq_length = cfg.max_tokens
    return emb


def _build_fastembed(cfg: EmbedConfig) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    return FastEmbedEmbeddings(
        model_name=cfg.model,
        max_length=cfg.max_tokens,
        batch_size=cfg.batch,
    )


def _build_hf_endpoint(cfg: EmbedConfig) -> Any:
    if not cfg.hf_token.strip():
        raise EmbeddingError("HF_TOKEN is required when EMBED_PROVIDER=hf_inference")
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model=cfg.model,
        provider="hf-inference",
        task="feature-extraction",
        huggingfacehub_api_token=cfg.hf_token,
    )


def _build_ollama(cfg: EmbedConfig, ollama_url: str) -> Any:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=cfg.model, base_url=ollama_url.rstrip("/"))


def validate_dims(cfg: EmbedConfig) -> None:
    mapped = mapping_embedding_dims()
    if mapped is not None and cfg.dims != mapped:
        raise EmbeddingError(f"EMBED_DIMS={cfg.dims} does not match the index mapping ({mapped})")


@lru_cache(maxsize=4)
def _cached(cfg: EmbedConfig, ollama_url: str) -> Embedder:
    validate_dims(cfg)
    q_prefix = QWEN3_QUERY_PREFIX if needs_query_prompt(cfg.model) else ""
    if cfg.provider == "ollama":
        return LangChainEmbedder(
            cfg,
            lambda: _build_ollama(cfg, ollama_url),
            batch_documents=True,
            query_prefix=q_prefix,
        )
    if cfg.provider == "hf_inference":
        return LangChainEmbedder(
            cfg,
            lambda: _build_hf_endpoint(cfg),
            batch_documents=True,
            query_prefix=q_prefix,
        )
    if cfg.provider == "fastembed":
        return LangChainEmbedder(
            cfg,
            lambda: _build_fastembed(cfg),
            query_prefix=q_prefix,
        )
    return LangChainEmbedder(cfg, lambda: _build_huggingface_local(cfg))


def get_embedder(settings: Settings) -> Embedder:
    """Cached singleton per config; the underlying LangChain Embeddings loads on first call."""
    return _cached(settings.embed, settings.llm.ollama_url)
