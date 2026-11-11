"""Search page (wireframe: Search). Two modes: Search (ES only) and Ask (ES + LLM)."""
from __future__ import annotations

import streamlit as st

from vaultlens.ask import AskService
from vaultlens.links import editor_uri, obsidian_uri
from vaultlens.llm.base import LLMError
from vaultlens.models import SearchSpec
from vaultlens.search.query_builder import compiled_query, inbound_links_body, more_like_this_body
from vaultlens.search.results import parse_response
from vaultlens.store import matches_saved
from vaultlens.ui import components, services

FACETS = {"type": "type", "folder": "folder.tree", "tags": "tags", "status": "status"}  # agg -> filter field
ss = st.session_state
cfg, be = services.settings(), services.backend()


def _clear() -> None:
    """Reset query, facet and custom-field filters, selection. Keeps Match/Sort/Size and the vault.

    Runs as an on_click callback: widget-bound session keys may only be changed here."""
    ss["query"] = ""
    ss["query_input"] = ""
    for agg in FACETS:
        ss[f"f_{agg}"] = []
    for key in ("selected_id", "loaded_saved", "confirm_delete"):
        ss.pop(key, None)
    components.clear_meta("sr")
    components.clear_date("sr")


st.title("Search")
mode = st.radio("Mode", ["Search", "Ask"], horizontal=True, label_visibility="collapsed")

ss.setdefault("query_input", ss.get("query", ""))
with st.form("query_form"):
    query = st.text_input("Query", key="query_input", placeholder="retrieval elasticsearch  (Enter to search)")
    c1, c2, c3 = st.columns(3)
    match = c1.selectbox("Match", ["all", "any", "phrase"], format_func=lambda m: {"all": "all words", "any": "any word", "phrase": "phrase"}[m])
    sort = c2.selectbox("Sort", ["relevance", "modified_desc", "created_desc", "title_asc"])
    size = c3.selectbox("Size", [10, 20, 50], index=1)
    b1, b2, _ = st.columns([1, 1, 6])
    submitted = b1.form_submit_button("Ask" if mode == "Ask" else "Search")
    b2.form_submit_button("Clear", on_click=_clear)
if submitted:
    ss["query"] = query

active = ss.get("active_vault", "All vaults")
# Filter widgets render later in the left column; like the facets, their values are read from
# session_state here so the spec is complete before the search runs.
spec = SearchSpec(
    query=ss.get("query", ""),
    vaults=[] if active == "All vaults" else [active],
    match_mode=match,
    sort=sort,
    size=size,
    filters={
        **{field: ss.get(f"f_{agg}", []) for agg, field in FACETS.items() if ss.get(f"f_{agg}")},
        **components.meta_filters_from_state("sr"),
    },
    date_range=components.date_range_from_state("sr"),
)

client, llm_err = services.llm()
if mode == "Ask":
    if client is None:
        st.info(llm_err or "Choose an LLM provider (ollama, gemini, or groq) in the sidebar to use Ask.")
    elif submitted and ss.get("query"):
        try:
            sources, stream = AskService(be.search, client).stream(ss["query"], spec)
            st.subheader("Answer")
            st.write_stream(stream)
            st.subheader("Sources")
            for s in sources:
                st.markdown(f"**[{s.n}]** {s.title} · `{s.path}`" + (f" › {s.heading}" if s.heading else ""))
        except LLMError as exc:
            st.error(str(exc))
        except Exception as exc:  # Elasticsearch down, mapping missing...
            st.error(f"Search failed: {type(exc).__name__}")
        st.divider()

try:
    resp = parse_response(be.search(spec))
except Exception as exc:
    st.error(f"Elasticsearch search failed ({type(exc).__name__}). Is the stack up and indexed?")
    st.stop()

# active filter summary + compiled query -------------------------------------------------------
if st.toggle("Show compiled query"):
    st.code(compiled_query(spec), language="json")

left, mid, right = st.columns([1, 2.4, 2])
with left:
    st.subheader("Filters")
    for agg in FACETS:
        buckets = resp.facets.get(agg, [])
        selected = ss.get(f"f_{agg}", [])
        options = list(dict.fromkeys([k for k, _ in buckets] + selected))
        counts = dict(buckets)
        st.multiselect(agg.title(), options, key=f"f_{agg}", format_func=lambda k, c=counts: f"{k} ({c.get(k, 0)})")
    components.meta_filter_editor("sr", None if active == "All vaults" else active)
    components.date_filter_widgets("sr")
    st.caption("Counts come from terms aggregations and update with every filter.")
    store = services.store()
    loaded = next((s for s in store.saved_searches() if s["name"] == ss.get("loaded_saved")), None)
    if loaded and matches_saved(loaded["spec"], spec.query, spec.filters):
        if ss.get("confirm_delete") == loaded["name"]:
            st.warning(f"Delete saved search '{loaded['name']}'?")
            d1, d2 = st.columns(2)
            if d1.button("Confirm delete", type="primary"):
                store.delete_search(loaded["name"])
                ss.pop("loaded_saved", None)
                ss.pop("confirm_delete", None)
                st.rerun()  # the sidebar list renders before this page
            if d2.button("Cancel"):
                ss.pop("confirm_delete", None)
                st.rerun()
        elif st.button("Delete search"):
            ss["confirm_delete"] = loaded["name"]
            st.rerun()
    elif ss.get("query") and st.button("Save search"):
        store.save_search(ss["query"], spec.to_dict())

with mid:
    st.caption(f"{resp.total:,} results · {resp.took_ms} ms")
    for hit in resp.hits:
        with st.container(border=True):
            if st.button(hit.title or hit.path, key=f"sel_{hit.doc_id}"):
                ss["selected_id"] = hit.doc_id
            st.caption(f"{hit.path} · {hit.type or '-'} · {', '.join(hit.tags[:4])}" + (f" · mod {hit.modified[:10]}" if hit.modified else ""))
            if hit.snippet:
                st.markdown(f"…{hit.snippet}…", unsafe_allow_html=True)  # snippet is html-encoded by ES

with right:
    chosen = next((h for h in resp.hits if h.doc_id == ss.get("selected_id")), None)
    if chosen is None:
        st.caption("Select a result to preview it.")
    else:
        st.subheader(chosen.title)
        st.caption(f"{chosen.vault} · {chosen.path}")
        vault_cfg = next((v for v in cfg.vaults if v.slug == chosen.vault), None)
        if vault_cfg:
            for label, uri in (("Open in Obsidian", obsidian_uri(vault_cfg, chosen.path)), ("Open in editor", editor_uri(vault_cfg, chosen.path))):
                if uri:
                    st.link_button(label, uri)
        st.markdown(f"**Frontmatter** · id `{chosen.note_id or '-'}` · type `{chosen.type or '-'}` · status `{chosen.status or '-'}` · modified `{chosen.modified or '-'}`")
        with st.expander("Dates"):
            st.markdown(f"Created `{chosen.created or '-'}`  \nModified `{chosen.modified or '-'}`  \nIndexed `{chosen.indexed_at or '-'}`")
        if chosen.chunks:
            st.markdown("**Best chunks**")
            for c in chosen.chunks:
                st.markdown(f"› {c.heading or '(intro)'}" + (f": …{c.snippet}…" if c.snippet else ""), unsafe_allow_html=True)
        try:
            inbound = be.raw_search(inbound_links_body(chosen.vault, chosen.path, chosen.title, chosen.aliases))
            st.markdown(f"**Links** · Out: {len(chosen.links_out)} · In: {inbound['hits']['total']['value']}")
            mlt = parse_response(be.raw_search(more_like_this_body(chosen.doc_id, chosen.vault)))
            if mlt.hits:
                st.markdown("**More like this**")
                for h in mlt.hits:
                    st.markdown(f"- {h.title} · `{h.path}`")
        except Exception:
            st.caption("Related notes unavailable.")
