"""Cached singletons for Streamlit. UI code gets everything through here."""
from __future__ import annotations

import streamlit as st

from ..config import Settings, load_settings
from ..indexing.service import IndexService
from ..llm.base import LLMClient, LLMError
from ..llm.factory import get_client
from ..search.backend import SearchBackend
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


@st.cache_resource
def index_service() -> IndexService:
    return IndexService(backend(), settings(), store())


def provider() -> str:
    return st.session_state.get("llm_provider", settings().llm.default_provider)


def llm() -> tuple[LLMClient | None, str | None]:
    """(client or None, error message or None). Provider 'off' returns (None, None)."""
    try:
        return get_client(provider(), settings().llm), None
    except LLMError as exc:
        return None, str(exc)
