"""Streamlit entrypoint: navigation + shared sidebar (vault, saved searches, index status, LLM)."""
from __future__ import annotations

import streamlit as st

from vaultlens.llm.base import GROQ_MODELS, OFF_DEVICE, PROVIDERS
from vaultlens.ui import components, services

st.set_page_config(page_title="Vault Search", page_icon="🔎", layout="wide")


def _load_saved(name: str, spec: dict) -> None:
    st.session_state["loaded_saved"] = name
    st.session_state.pop("confirm_delete", None)
    st.session_state["query"] = spec.get("query", "")
    st.session_state["query_input"] = spec.get("query", "")
    for agg, field in (("type", "type"), ("folder", "folder.tree"), ("tags", "tags"), ("status", "status")):
        st.session_state[f"f_{agg}"] = spec.get("filters", {}).get(field, [])
    components.set_meta_rows("sr", spec.get("filters", {}))
    components.set_date_range("sr", spec.get("date_range"))


def sidebar() -> None:
    cfg = services.settings()
    be = services.backend()
    with st.sidebar:
        st.header("Vault Search")
        slugs = ["All vaults"] + [v.slug for v in cfg.vaults]
        st.selectbox("Active vault", slugs, key="active_vault")

        st.subheader("Saved searches")
        saved = services.store().saved_searches()
        for item in saved:
            st.button(item["name"], key=f"saved_{item['name']}", on_click=_load_saved, args=(item["name"], item["spec"]))
        if not saved:
            st.caption("None yet. Save one from the Search page.")

        st.subheader("Index")
        ok, state = be.health()
        if ok:
            stats = be.stats()
            st.caption(f"Elasticsearch {state} · {stats['docs']:,} notes · {stats['chunks']:,} chunks")
        else:
            st.error(f"Elasticsearch unreachable ({state})")

        st.subheader("LLM")
        st.radio(
            "Provider", PROVIDERS, key="llm_provider",
            index=PROVIDERS.index(cfg.llm.default_provider) if cfg.llm.default_provider in PROVIDERS else 0,
            help="'off' = plain Elasticsearch search. Ask mode needs ollama, gemini, or groq.",
        )
        prov = services.provider()
        if prov == "groq":
            st.radio(
                "Groq model",
                GROQ_MODELS,
                key="groq_model",
                index=GROQ_MODELS.index(cfg.llm.groq_model) if cfg.llm.groq_model in GROQ_MODELS else 0,
                help="Auto-falls back to the other Groq model on HTTP 429 rate limits.",
            )
        if prov in OFF_DEVICE:
            api_label = "Groq's API" if prov == "groq" else "Google's API"
            st.warning(f"Ask mode sends excerpts of matching notes to {api_label}.")
        client, err = services.llm()
        if err:
            st.error(err)
        elif client:
            good, msg = client.health()
            (st.success if good else st.warning)(f"{client.provider} · {client.model}: {msg}")


sidebar()
pages = [
    st.Page("views/search.py", title="Search", default=True),
    st.Page("views/workbench.py", title="Workbench"),
    st.Page("views/indexing.py", title="Vaults and Indexing"),
    st.Page("views/evaluate.py", title="Evaluate"),
]
st.navigation(pages).run()
