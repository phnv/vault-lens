"""Workbench page (wireframe: Workbench). Every control maps to one SearchSpec field."""
from __future__ import annotations

import json

import streamlit as st

from vaultlens.models import SearchSpec
from vaultlens.search.query_builder import ALLOWED_MATCH_TYPES, ALLOWED_QUERY_FIELDS, compiled_query
from vaultlens.search.results import parse_response
from vaultlens.ui import components, services

be = services.backend()
st.title("Workbench")
level = st.radio("Level", ["Everyday", "Advanced"], horizontal=True)

def _clear() -> None:
    """Reset the query and custom-field filters (on_click callback); advanced options stay."""
    st.session_state["wb_query"] = ""
    components.clear_meta("wb")


spec = SearchSpec(query=st.text_input("Query", key="wb_query"), explain=False)
st.button("Clear", on_click=_clear, key="wb_clear")
with st.expander("Custom fields", expanded=bool(st.session_state.get("wb_meta_rows"))):
    components.meta_filter_editor("wb", None)
spec.filters = components.meta_filters_from_state("wb")
spec.match_mode = st.selectbox("Match mode", ["all", "any", "phrase"])  # type: ignore[assignment]

if level == "Advanced":
    spec.match_type = st.selectbox("Match type", sorted(ALLOWED_MATCH_TYPES), index=sorted(ALLOWED_MATCH_TYPES).index("best_fields"))
    chosen = st.multiselect("Fields and boosts", sorted(ALLOWED_QUERY_FIELDS), default=list(spec.fields))
    spec.fields = {f: st.number_input(f"{f} boost", 0.0, 10.0, spec.fields.get(f, 1.0), 0.5, key=f"boost_{f}") for f in chosen}
    spec.fuzziness = st.selectbox("Fuzziness", [None, "AUTO", "1", "2"])
    spec.minimum_should_match = st.text_input("minimum_should_match", "") or None
    spec.explain = st.toggle("explain")
    spec.profile = st.toggle("profile", value=True)
    spec.with_chunks = st.toggle("nested chunk inner_hits", value=True)
spec.facets = False

tab_results, tab_dsl, tab_explain, tab_profile, tab_analyzer = st.tabs(["Results", "Query DSL", "Explain", "Profile", "Analyzer"])
with tab_dsl:
    st.code(compiled_query(spec), language="json")

if st.button("Run", type="primary"):
    try:
        resp = parse_response(be.search(spec))
    except Exception as exc:
        st.error(f"Search failed: {type(exc).__name__}")
        st.stop()
    with tab_results:
        st.caption(f"{resp.total:,} results · {resp.took_ms} ms")
        for h in resp.hits:
            st.markdown(f"**{h.title}** · score {h.score:.3f} · `{h.path}`" if h.score is not None else f"**{h.title}** · `{h.path}`")
            for c in h.chunks:
                snippet = f": …{c.snippet}…" if c.snippet else ""  # html-encoded by ES
                st.markdown(f"› {c.heading or '(intro)'}{snippet}", unsafe_allow_html=True)
    with tab_explain:
        for h in resp.hits[:5]:
            if h.explanation:
                st.markdown(f"**{h.title}**")
                st.json(h.explanation, expanded=False)
        if not any(h.explanation for h in resp.hits):
            st.caption("Enable 'explain' at Advanced level.")
    with tab_profile:
        st.json(resp.raw.get("profile", {"info": "Enable 'profile' at Advanced level."}), expanded=False)

with tab_analyzer:
    if not spec.query.strip():
        st.caption("Type a query to see token streams.")
    else:
        for name in ("standard", "std_folded", "pt_folded", "en_folded"):
            try:
                st.markdown(f"**{name}**: `{' | '.join(be.analyze(name, spec.query))}`")
            except Exception as exc:
                st.caption(f"{name}: unavailable ({type(exc).__name__})")
