"""Search page: Simple, Ask AI, Hybrid, AI Search, and Agentic tiers (ADR-0009)."""

from __future__ import annotations

import streamlit as st

from vaultlens.ask import AskService
from vaultlens.links import editor_uri, obsidian_uri
from vaultlens.llm.base import LLMError
from vaultlens.models import SearchResponse, SearchSpec, TraceEvent
from vaultlens.search.query_builder import compiled_query, inbound_links_body, more_like_this_body
from vaultlens.search.results import parse_response
from vaultlens.search.tiers import TIERS, get_tier
from vaultlens.store import matches_saved
from vaultlens.ui import components, services

FACETS = {
    "type": "type",
    "folder": "folder.tree",
    "tags": "tags",
    "status": "status",
}  # agg -> filter field
ss = st.session_state
cfg, be = services.settings(), services.backend()


def _clear() -> None:
    """Reset query, facet and custom-field filters, selection. Keeps Match/Sort/Size and the vault.

    Runs as an on_click callback: widget-bound session keys may only be changed here."""
    ss["query"] = ""
    ss["query_input"] = ""
    for agg in FACETS:
        ss[f"f_{agg}"] = []
    for key in (
        "selected_id",
        "loaded_saved",
        "confirm_delete",
        "last_agent_trace",
        "last_agent_warnings",
    ):
        ss.pop(key, None)
    components.clear_meta("sr")
    components.clear_date("sr")


def _render_trace_row(ev: TraceEvent, show_prompts: bool = False) -> None:
    lat = f" · `{ev.ms} ms`" if ev.ms is not None else ""
    st.markdown(f"- **{ev.step}** (`{ev.kind}`){lat} — {ev.summary}")
    if ev.data:
        detail = {k: v for k, v in ev.data.items() if show_prompts or k != "prompt"}
        if detail:
            with st.expander(f"Details: {ev.step}", expanded=False):
                st.json(detail)


def _render_agent_log(events: list[TraceEvent], expanded: bool = False) -> None:
    if not events:
        return
    with st.expander("Agent log", expanded=expanded):
        show_prompts = st.toggle("Show prompts", value=False, key="agent_log_show_prompts")
        for ev in events:
            _render_trace_row(ev, show_prompts=show_prompts)


st.title("Search")
tier_ids = [t.id for t in TIERS]
tier_captions = [t.caption for t in TIERS]
selected_tier_id = st.radio(
    "Mode",
    tier_ids,
    format_func=lambda tid: get_tier(tid).label,
    captions=tier_captions,
    horizontal=True,
    label_visibility="collapsed",
)
tier = get_tier(selected_tier_id)

ss.setdefault("query_input", ss.get("query", ""))
if ss.get("match_mode") not in ("any", "all", "phrase"):
    ss["match_mode"] = "any"
with st.form("query_form"):
    query = st.text_input(
        "Query", key="query_input", placeholder="retrieval elasticsearch  (Enter to search)"
    )
    c1, c2, c3 = st.columns(3)
    match = c1.selectbox(
        "Match",
        ["any", "all", "phrase"],
        key="match_mode",
        format_func=lambda m: {"all": "all words", "any": "any word", "phrase": "phrase"}[m],
    )
    sort = c2.selectbox("Sort", ["relevance", "modified_desc", "created_desc", "title_asc"])
    size = c3.selectbox("Size", [10, 20, 50], index=1)
    b1, b2, _ = st.columns([1, 1, 6])
    btn_label = "Ask" if tier.needs_llm else "Search"
    submitted = b1.form_submit_button(btn_label)
    b2.form_submit_button("Clear", on_click=_clear)
if submitted:
    ss["query"] = query

active = ss.get("active_vault", "All vaults")
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

tier_ok, tier_reason = services.tier_status(tier)
resp: SearchResponse | None = None

if tier.id == "ask":
    client, llm_err = services.llm()
    if client is None:
        st.info(
            llm_err or "Choose an LLM provider (ollama, gemini, or groq) in the sidebar to use Ask."
        )
    elif submitted and ss.get("query"):
        try:
            sources, stream = AskService(be.search, client).stream(ss["query"], spec)
            st.subheader("Answer")
            st.write_stream(stream)
            st.subheader("Sources")
            for s in sources:
                st.markdown(
                    f"**[{s.n}]** {s.title} · `{s.path}`" + (f" › {s.heading}" if s.heading else "")
                )
        except LLMError as exc:
            st.error(str(exc))
        except Exception as exc:  # Elasticsearch down, mapping missing...
            st.error(f"Search failed: {type(exc).__name__}")
        st.divider()

elif tier.id == "hybrid":
    if not tier_ok:
        st.info(tier_reason)
    else:
        try:
            trace_events: list[TraceEvent] = []
            hresp = services.hybrid(trace=trace_events.append).search(
                ss.get("query", ""), spec, top_n=size
            )
            for w in hresp.warnings:
                st.warning(w)
            resp = SearchResponse(
                total=hresp.total,
                took_ms=hresp.took_ms,
                hits=hresp.hits,
                facets=hresp.facets,
            )
        except Exception as exc:
            st.error(f"Hybrid search failed ({type(exc).__name__}).")

elif tier.id in ("ai_search", "agentic"):
    if not tier_ok:
        st.info(tier_reason)
    elif submitted and ss.get("query"):
        from vaultlens.agents.agentic import run_agentic_search
        from vaultlens.agents.ai_search import run_ai_search
        from vaultlens.agents.llm import get_chat_model
        from vaultlens.agents.trace import TraceSink

        try:
            chat_model = get_chat_model(
                services.provider(), cfg.llm, groq_model=services.groq_model()
            )
            with st.status("Agent log", expanded=True) as status_box:

                def _on_live_event(ev: TraceEvent) -> None:
                    lat = f" ({ev.ms} ms)" if ev.ms is not None else ""
                    status_box.write(f"**{ev.step}** [`{ev.kind}`]{lat}: {ev.summary}")

                sink = TraceSink(on_event=_on_live_event)
                hsearcher = services.hybrid(trace=sink)
                if tier.id == "ai_search":
                    out = run_ai_search(
                        ss["query"],
                        spec,
                        chat_model,
                        hsearcher,
                        sink=sink,
                        show_prompts=True,
                    )
                else:
                    # Fetch live facet values to validate any LLM-proposed filters
                    base_facets: dict[str, list[str]] = {}
                    try:
                        f_raw = parse_response(
                            be.search(SearchSpec(vaults=spec.vaults, size=0, facets=True))
                        ).facets
                        base_facets = {
                            k: [val for val, _ in buckets] for k, buckets in f_raw.items()
                        }
                    except Exception:
                        base_facets = {}
                    out = run_agentic_search(
                        ss["query"],
                        spec,
                        chat_model,
                        hsearcher,
                        cfg=cfg.agent,
                        allowed_facets=base_facets,
                        sink=sink,
                        show_prompts=True,
                    )
                status_box.update(label="Agent run complete", state="complete", expanded=False)

            for w in out.warnings:
                st.warning(w)
            if out.answer:
                st.subheader("Answer")
                st.markdown(out.answer)
            if out.sources:
                st.subheader("Sources")
                for s in out.sources:
                    st.markdown(
                        f"**[{s.n}]** {s.title} · `{s.path}`"
                        + (f" › {s.heading}" if s.heading else "")
                    )
            _render_agent_log(sink.events, expanded=False)
            st.divider()
            resp = SearchResponse(
                total=out.primary.total,
                took_ms=out.primary.took_ms,
                hits=out.primary.hits,
                facets=out.primary.facets,
            )
        except LLMError as exc:
            st.warning(f"{exc} — showing hybrid results instead.")
        except Exception as exc:
            st.error(f"Agent search failed ({type(exc).__name__}).")

if resp is None:
    try:
        resp = parse_response(be.search(spec))
    except Exception as exc:
        st.error(
            f"Elasticsearch search failed ({type(exc).__name__}). Is the stack up and indexed?"
        )
        st.stop()

# active filter summary + compiled query -------------------------------------------------------
if st.toggle("Show compiled query"):
    st.code(compiled_query(spec), language="json")
    if tier.needs_embeddings and resp.hits:
        scores_preview = [
            {"doc_id": h.doc_id, "title": h.title, "score": h.score} for h in resp.hits[:10]
        ]
        st.json({"hybrid_top_scores": scores_preview})

left, mid, right = st.columns([1, 2.4, 2])
with left:
    st.subheader("Filters")
    for agg in FACETS:
        buckets = resp.facets.get(agg, [])
        selected = ss.get(f"f_{agg}", [])
        options = list(dict.fromkeys([k for k, _ in buckets] + selected))
        counts = dict(buckets)
        st.multiselect(
            agg.title(),
            options,
            key=f"f_{agg}",
            format_func=lambda k, c=counts: f"{k} ({c.get(k, 0)})",
        )
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
            st.caption(
                f"{hit.path} · {hit.type or '-'} · {', '.join(hit.tags[:4])}"
                + (f" · mod {hit.modified[:10]}" if hit.modified else "")
            )
            if hit.snippet:
                st.markdown(
                    f"…{hit.snippet}…", unsafe_allow_html=True
                )  # snippet is html-encoded by ES

with right:
    chosen = next((h for h in resp.hits if h.doc_id == ss.get("selected_id")), None)
    if chosen is None:
        st.caption("Select a result to preview it.")
    else:
        st.subheader(chosen.title)
        st.caption(f"{chosen.vault} · {chosen.path}")
        vault_cfg = next((v for v in cfg.vaults if v.slug == chosen.vault), None)
        if vault_cfg:
            for label, uri in (
                ("Open in Obsidian", obsidian_uri(vault_cfg, chosen.path)),
                ("Open in editor", editor_uri(vault_cfg, chosen.path)),
            ):
                if uri:
                    st.link_button(label, uri)
        st.markdown(
            f"**Frontmatter** · id `{chosen.note_id or '-'}` · type `{chosen.type or '-'}` · "
            f"status `{chosen.status or '-'}` · modified `{chosen.modified or '-'}`"
        )
        with st.expander("Dates"):
            st.markdown(
                f"Created `{chosen.created or '-'}`  \nModified `{chosen.modified or '-'}`  \n"
                f"Indexed `{chosen.indexed_at or '-'}`"
            )
        if chosen.chunks:
            st.markdown("**Best chunks**")
            for c in chosen.chunks:
                st.markdown(
                    f"› {c.heading or '(intro)'}" + (f": …{c.snippet}…" if c.snippet else ""),
                    unsafe_allow_html=True,
                )
        try:
            inbound = be.raw_search(
                inbound_links_body(chosen.vault, chosen.path, chosen.title, chosen.aliases)
            )
            st.markdown(
                f"**Links** · Out: {len(chosen.links_out)} · "
                f"In: {inbound['hits']['total']['value']}"
            )
            mlt = parse_response(be.raw_search(more_like_this_body(chosen.doc_id, chosen.vault)))
            if mlt.hits:
                st.markdown("**More like this**")
                for h in mlt.hits:
                    st.markdown(f"- {h.title} · `{h.path}`")
        except Exception:
            st.caption("Related notes unavailable.")
