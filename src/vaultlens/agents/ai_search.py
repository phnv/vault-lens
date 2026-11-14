"""Tier b (`ai_search`) LangGraph StateGraph (ADR-0009 section 6.9).

Fixed pipeline: `START -> transform -> retrieve -> answer -> END`.
Budget: 2 LLM calls (plus at most 1 repair retry on `transform`).
"""

from __future__ import annotations

import operator
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..models import HybridResponse, Passage, SearchSpec, TraceEvent
from ..search.hybrid import HybridSearcher, rrf_fuse
from ..semantic.reranker import prepare_rerank_text
from .llm import invoke_structured_with_repair
from .prompts import (
    NO_RESULTS,
    SYSTEM_PROMPT,
    TRANSFORM_SYSTEM_PROMPT,
    Source,
    build_answer_user_prompt,
    build_passages_context,
)
from .schemas import QueryPlan
from .trace import TraceSink, now_iso

MAX_QUERIES = 6


def dedupe_queries(
    original: str,
    rewrites: Sequence[str],
    sub_questions: Sequence[str],
    cap: int = MAX_QUERIES,
) -> list[str]:
    """Return `[original, ...]` de-duplicated case/whitespace-insensitively, capped at `cap`."""
    out: list[str] = []
    seen: set[str] = set()
    for q in [original, *list(rewrites)[:3], *list(sub_questions)[:4]]:
        clean = " ".join(q.split())
        if not clean:
            continue
        norm = clean.casefold()
        if norm not in seen:
            seen.add(norm)
            out.append(clean)
        if len(out) >= cap:
            break
    return out or [original.strip()]


def merge_and_rerank_passages(
    question: str,
    per_query: dict[str, HybridResponse],
    searcher: HybridSearcher,
    top_n: int = 8,
) -> list[Passage]:
    """Merge passages across queries via RRF over note ids, then rerank against `question`."""
    rankings = [[h.doc_id for h in resp.hits] for resp in per_query.values() if resp.hits]
    fused_notes = rrf_fuse(rankings, k=searcher.cfg.rrf_k) if rankings else []
    note_rrf = dict(fused_notes)

    passages_by_key: dict[str, Passage] = {}
    for resp in per_query.values():
        for p in resp.passages:
            pkey = f"{p.doc_id}#{p.position}"
            if pkey not in passages_by_key:
                passages_by_key[pkey] = replace(p, rrf_score=note_rrf.get(p.doc_id, p.rrf_score))

    if not passages_by_key:
        return []

    note_rank = {doc_id: idx for idx, (doc_id, _) in enumerate(fused_notes)}
    initial_keys = sorted(
        passages_by_key,
        key=lambda pk: (
            note_rank.get(passages_by_key[pk].doc_id, 9999),
            passages_by_key[pk].position,
            pk,
        ),
    )

    if searcher.rerank_cfg.enabled and searcher.reranker is not None:
        try:
            texts = [
                prepare_rerank_text(
                    question,
                    passages_by_key[pk].heading,
                    passages_by_key[pk].text,
                    oversized_chars=searcher.oversized_chars,
                )
                for pk in initial_keys
            ]
            scores = searcher.reranker.score_pairs(question, texts)
            for pk, sc in zip(initial_keys, scores, strict=True):
                passages_by_key[pk].rerank_score = float(sc)
            rerank_order = [
                pk
                for pk, _ in sorted(
                    zip(initial_keys, scores, strict=True),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ]
            if searcher.rerank_cfg.mode == "replace":
                final_keys = rerank_order
            else:
                blended = rrf_fuse([initial_keys, rerank_order], k=searcher.cfg.rrf_k)
                for pk, bscore in blended:
                    passages_by_key[pk].rrf_score = bscore
                final_keys = [pk for pk, _ in blended]
            return [passages_by_key[pk] for pk in final_keys[:top_n]]
        except Exception:
            pass

    return [passages_by_key[pk] for pk in initial_keys[:top_n]]


def _extract_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


class AISearchState(TypedDict, total=False):
    question: str
    spec: SearchSpec
    queries: list[str]
    plan: dict[str, Any]
    per_query: dict[str, HybridResponse]
    merged_passages: list[Passage]
    primary: HybridResponse
    answer: str
    sources: list[Source]
    trace: Annotated[list[TraceEvent], operator.add]
    warnings: Annotated[list[str], operator.add]
    llm_calls: int


@dataclass
class AISearchOutput:
    answer: str
    sources: list[Source]
    primary: HybridResponse
    merged_passages: list[Passage]
    queries: list[str]
    trace: list[TraceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0


def build_ai_search_graph(
    chat_model: Any,
    searcher: HybridSearcher,
    *,
    top_n: int = 8,
    sink: TraceSink | None = None,
    show_prompts: bool = False,
):
    def _record(
        step: str,
        kind: str,
        summary: str,
        ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        ev = TraceEvent(
            ts=now_iso(), step=step, kind=kind, summary=summary, ms=ms, data=dict(data or {})
        )
        if sink is not None:
            sink(ev)
        return ev

    def transform_node(state: AISearchState) -> dict[str, Any]:
        q = state["question"].strip()
        t0 = time.monotonic()
        warnings: list[str] = []
        try:
            plan_obj, calls = invoke_structured_with_repair(
                chat_model, QueryPlan, TRANSFORM_SYSTEM_PROMPT, f"Question: {q}"
            )
            rewrites = list(plan_obj.rewrites)[:3]
            sub_qs = list(plan_obj.sub_questions)[:4]
            queries = dedupe_queries(q, rewrites, sub_qs, cap=MAX_QUERIES)
            ms = int((time.monotonic() - t0) * 1000)
            data: dict[str, Any] = {
                "rewrites": rewrites,
                "sub_questions": sub_qs,
                "queries": queries,
                "llm_calls": calls,
            }
            if show_prompts:
                data["prompt"] = f"Question: {q}"
            ev = _record(
                "transform",
                "llm",
                f"Expanded into {len(queries)} queries ({len(sub_qs)} sub-questions)",
                ms=ms,
                data=data,
            )
            return {
                "queries": queries,
                "plan": {"rewrites": rewrites, "sub_questions": sub_qs},
                "llm_calls": state.get("llm_calls", 0) + calls,
                "trace": [ev],
                "warnings": warnings,
            }
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            msg = (
                f"Query expansion failed ({type(exc).__name__}); "
                "continuing with original question only."
            )
            warnings.append(msg)
            ev = _record(
                "transform",
                "warning",
                msg,
                ms=ms,
                data={"queries": [q], "error": type(exc).__name__},
            )
            return {
                "queries": [q],
                "plan": {"rewrites": [], "sub_questions": []},
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [ev],
                "warnings": warnings,
            }

    def retrieve_node(state: AISearchState) -> dict[str, Any]:
        q_orig = state["question"].strip()
        queries = state.get("queries") or [q_orig]
        spec = state.get("spec") or SearchSpec()
        user_filters = {
            k: {"values": list(v), "source": "user"} for k, v in spec.filters.items() if v
        }
        t0 = time.monotonic()
        per_query: dict[str, HybridResponse] = {}
        warnings: list[str] = []

        for q in queries:
            resp = searcher.search(q, spec, top_n=top_n)
            per_query[q] = resp
            warnings.extend(resp.warnings)

        primary = per_query.get(queries[0]) or HybridResponse(passages=[], hits=[])
        merged = merge_and_rerank_passages(q_orig, per_query, searcher, top_n=top_n)
        ms = int((time.monotonic() - t0) * 1000)
        ev = _record(
            "retrieve",
            "search",
            f"Ran {len(queries)} hybrid searches -> {len(merged)} merged passages",
            ms=ms,
            data={
                "queries": queries,
                "filters": user_filters,
                "per_query_legs": {q: r.legs for q, r in per_query.items()},
                "merged_passages": [
                    {"doc_id": p.doc_id, "title": p.title, "path": p.path, "heading": p.heading}
                    for p in merged
                ],
            },
        )
        return {
            "per_query": per_query,
            "primary": primary,
            "merged_passages": merged,
            "trace": [ev],
            "warnings": warnings,
        }

    def answer_node(state: AISearchState) -> dict[str, Any]:
        passages = state.get("merged_passages") or []
        q = state["question"].strip()
        sub_qs = (state.get("plan") or {}).get("sub_questions") or []
        context, sources = build_passages_context(passages)
        if not sources:
            ev = _record(
                "answer",
                "decision",
                "No passages retrieved; skipped answer LLM call.",
                ms=0,
                data={"sources": 0},
            )
            return {"answer": NO_RESULTS, "sources": [], "trace": [ev]}

        user_prompt = build_answer_user_prompt(q, context, sub_questions=sub_qs)
        t0 = time.monotonic()
        try:
            msg = chat_model.invoke([("system", SYSTEM_PROMPT), ("human", user_prompt)])
            text = _extract_text(msg).strip() or NO_RESULTS
            ms = int((time.monotonic() - t0) * 1000)
            data: dict[str, Any] = {
                "sources": [
                    {"n": s.n, "title": s.title, "path": s.path, "heading": s.heading}
                    for s in sources
                ],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }
            if show_prompts:
                data["prompt"] = user_prompt
            ev = _record(
                "answer",
                "llm",
                f"Synthesized cited answer from {len(sources)} passages",
                ms=ms,
                data=data,
            )
            return {
                "answer": text,
                "sources": sources,
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [ev],
            }
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            warn = (
                f"Answer generation failed ({type(exc).__name__}); "
                "returning retrieved hybrid results."
            )
            ev = _record("answer", "warning", warn, ms=ms, data={"error": type(exc).__name__})
            return {
                "answer": "",
                "sources": sources,
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [ev],
                "warnings": [warn],
            }

    g = StateGraph(AISearchState)
    g.add_node("transform", transform_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_edge(START, "transform")
    g.add_edge("transform", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


def run_ai_search(
    question: str,
    spec: SearchSpec,
    chat_model: Any,
    searcher: HybridSearcher,
    *,
    top_n: int = 8,
    sink: TraceSink | None = None,
    show_prompts: bool = False,
) -> AISearchOutput:
    graph = build_ai_search_graph(
        chat_model, searcher, top_n=top_n, sink=sink, show_prompts=show_prompts
    )
    final: AISearchState = graph.invoke(
        {"question": question, "spec": spec, "trace": [], "warnings": [], "llm_calls": 0}
    )
    return AISearchOutput(
        answer=final.get("answer", ""),
        sources=final.get("sources", []),
        primary=final.get("primary") or HybridResponse(passages=[], hits=[]),
        merged_passages=final.get("merged_passages", []),
        queries=final.get("queries", [question]),
        trace=final.get("trace", []),
        warnings=final.get("warnings", []),
        llm_calls=final.get("llm_calls", 0),
    )
