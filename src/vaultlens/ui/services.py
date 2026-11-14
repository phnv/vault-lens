"""Cached singletons for Streamlit. UI code gets everything through here."""

from __future__ import annotations

import streamlit as st

from ..config import Settings, load_settings
from ..indexing.service import IndexService
from ..llm.base import LLMClient, LLMError
from ..llm.factory import get_client
from ..models import TraceEvent
from ..search.backend import SearchBackend
from ..search.hybrid import HybridSearcher, TraceCallback
from ..search.query_builder import meta_values_body
from ..search.results import parse_response
from ..search.tiers import Tier, evaluate_tier_status
from ..semantic.embedder import Embedder, get_embedder
from ..semantic.reranker import Reranker, get_reranker
from ..store import JsonStore


@st.cache_resource
def settings() -> Settings:
    return load_settings()


@st.cache_resource
def backend() -> SearchBackend:
    s = settings()
    return SearchBackend(s.es_url, s.es_alias)


@st.cache_resource
def store() -> JsonStore:
    return JsonStore(settings().data_dir)


@st.cache_data(ttl=60, show_spinner=False)
def meta_values(key: str, vault: str | None) -> list[tuple[str, int]]:
    """Distinct (value, count) for a custom field; short TTL so a sync shows up quickly."""
    raw = backend().raw_search(meta_values_body(key, vault))
    return parse_response(raw).facets.get("values", [])


@st.cache_resource
def embedder() -> Embedder:
    return get_embedder(settings())


@st.cache_resource
def reranker() -> Reranker | None:
    return get_reranker(settings())


@st.cache_resource
def index_service() -> IndexService:
    s = settings()
    emb = embedder() if s.embeddings_enabled else None
    return IndexService(backend(), s, store(), embedder=emb)


def hybrid(trace: TraceCallback | None = None) -> HybridSearcher:
    s = settings()
    return HybridSearcher(
        backend().raw_search,
        embedder(),
        reranker(),
        cfg=s.hybrid,
        rerank_cfg=s.rerank,
        trace=trace,
        oversized_chars=max(1600, s.chunk_max_chars),
    )


def provider() -> str:
    return st.session_state.get("llm_provider", settings().llm.default_provider)


def groq_model() -> str:
    return st.session_state.get("groq_model", settings().llm.groq_model)


def llm() -> tuple[LLMClient | None, str | None]:
    """(client or None, error message or None). Provider 'off' returns (None, None)."""
    try:
        return get_client(provider(), settings().llm, groq_model=groq_model()), None
    except LLMError as exc:
        return None, str(exc)


def tier_status(tier: Tier) -> tuple[bool, str]:
    """(ok, reason) for `tier` against live settings and index _meta."""
    s = settings()
    _, llm_err = llm()
    live_emb = "off"
    total_chunks: int | None = None
    embedded_chunks: int | None = None
    if tier.needs_embeddings:
        try:
            st = backend().stats()
            live_emb = str(st.get("meta", {}).get("embedding_model", "off"))
            total_chunks = int(st.get("chunks", 0))
            embedded_chunks = int(st.get("embedded_chunks", 0))
        except Exception:
            live_emb = "off"
    return evaluate_tier_status(
        tier,
        llm_provider=provider(),
        llm_error=llm_err,
        config_embedding_model=s.embedding_model,
        live_embedding_model=live_emb,
        total_chunks=total_chunks,
        embedded_chunks=embedded_chunks,
    )


__all__ = [
    "TraceEvent",
    "backend",
    "embedder",
    "groq_model",
    "hybrid",
    "index_service",
    "llm",
    "meta_values",
    "provider",
    "reranker",
    "settings",
    "store",
    "tier_status",
]
