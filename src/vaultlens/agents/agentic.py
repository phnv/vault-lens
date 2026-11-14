"""Tier c (`agentic`) LangGraph StateGraph (ADR-0009 section 6.3 & 6.10).

Graph: `START -> plan -> retrieve -> grade -> (sufficient/budget -> synthesize | refine)`.
Enforces hard budgets (`max_iterations`, `max_llm_calls`, `timeout_s`, evidence cap, recursion),
filter validation against live facets, and deterministic filter auto-relaxation (< `min_hits`).
"""

from __future__ import annotations

import operator
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import AgentConfig
from ..models import HybridResponse, Passage, SearchSpec, TraceEvent
from ..search.hybrid import HybridSearcher
from ..search.query_builder import ALLOWED_FILTER_FIELDS
from .ai_search import MAX_QUERIES, _extract_text, dedupe_queries, merge_and_rerank_passages
from .llm import invoke_structured_with_repair
from .prompts import (
    GRADE_SYSTEM_PROMPT,
    NO_RESULTS,
    PLAN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    Source,
    build_answer_user_prompt,
    build_grade_user_prompt,
    build_passages_context,
)
from .schemas import AgentPlan, Grade
from .trace import TraceSink, now_iso

AGENT_ALLOWED_FILTER_FIELDS = ALLOWED_FILTER_FIELDS - {"links_out"}
FACET_KEY_TO_FILTER_FIELD = {
    "type": "type",
    "status": "status",
    "tags": "tags",
    "folder": "folder.tree",
    "folder.tree": "folder.tree",
    "vault": "vault",
}
MAX_EVIDENCE_PASSAGES = 24


def validate_agent_filters(
    proposed: Mapping[str, Sequence[str]] | None,
    allowed_facets: Mapping[str, Sequence[str]],
    user_filters: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Validate LLM-proposed filters against `AGENT_ALLOWED_FILTER_FIELDS` and live facet values.

    - User filters are never overridden or modified here.
    - Anything outside allowed fields or live facets is dropped and recorded in `dropped`.
    """
    if not proposed:
        return {}, []
    valid: dict[str, list[str]] = {}
    dropped: list[str] = []

    for raw_field, raw_vals in proposed.items():
        if not raw_vals:
            continue
        mapped_field = FACET_KEY_TO_FILTER_FIELD.get(raw_field, raw_field)
        if mapped_field not in AGENT_ALLOWED_FILTER_FIELDS:
            dropped.append(f"disallowed field {raw_field!r}")
            continue
        if mapped_field in user_filters and user_filters[mapped_field]:
            dropped.append(f"field {mapped_field!r} already set by user")
            continue

        facet_lookup_key = "folder" if mapped_field == "folder.tree" else mapped_field
        allowed_vals = set(
            allowed_facets.get(facet_lookup_key) or allowed_facets.get(mapped_field) or ()
        )
        kept: list[str] = []
        for val in raw_vals:
            sval = str(val).strip()
            if not sval:
                continue
            if allowed_vals and sval not in allowed_vals:
                dropped.append(f"{mapped_field}={sval!r} not in live facets")
                continue
            if sval not in kept:
                kept.append(sval)
        if kept:
            valid[mapped_field] = kept
    return valid, dropped


def merge_user_and_agent_filters(
    user_filters: Mapping[str, Sequence[str]],
    agent_filters: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Combine user filters (authoritative) with agent filters."""
    out: dict[str, list[str]] = {k: list(v) for k, v in user_filters.items() if v}
    for k, v in agent_filters.items():
        if v and k not in out:
            out[k] = list(v)
    return out


class AgenticState(TypedDict, total=False):
    question: str
    spec: SearchSpec
    queries: list[str]
    issued_queries: list[str]
    open_questions: list[str]
    agent_filters: dict[str, list[str]]
    evidence: list[Passage]
    primary: HybridResponse
    grade: dict[str, Any]
    iteration: int
    llm_calls: int
    started_at: float
    answer: str
    sources: list[Source]
    trace: Annotated[list[TraceEvent], operator.add]
    warnings: Annotated[list[str], operator.add]


@dataclass
class AgenticOutput:
    answer: str
    sources: list[Source]
    primary: HybridResponse
    evidence: list[Passage]
    queries: list[str]
    trace: list[TraceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    iterations: int = 0


def build_agentic_graph(
    chat_model: Any,
    searcher: HybridSearcher,
    *,
    cfg: AgentConfig | None = None,
    allowed_facets: Mapping[str, Sequence[str]] | None = None,
    top_n: int = 8,
    sink: TraceSink | None = None,
    show_prompts: bool = False,
):
    agent_cfg = cfg or AgentConfig()
    facets_map: dict[str, list[str]] = {k: list(v)[:20] for k, v in (allowed_facets or {}).items()}

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

    def plan_node(state: AgenticState) -> dict[str, Any]:
        q = state["question"].strip()
        spec = state.get("spec") or SearchSpec()
        t0 = time.monotonic()
        warnings: list[str] = []

        facets_desc = (
            "\n".join(
                f"- {k}: {', '.join(vals[:15])}" for k, vals in sorted(facets_map.items()) if vals
            )
            or "(none)"
        )
        user_prompt = f"Question: {q}\n\nAllowed facet values:\n{facets_desc}"

        try:
            plan_obj, calls = invoke_structured_with_repair(
                chat_model, AgentPlan, PLAN_SYSTEM_PROMPT, user_prompt
            )
            rewrites = list(plan_obj.rewrites)[:3]
            sub_qs = list(plan_obj.sub_questions)[:4]
            queries = dedupe_queries(q, rewrites, sub_qs, cap=MAX_QUERIES)
            valid_filters, dropped = validate_agent_filters(
                plan_obj.filters, facets_map, spec.filters
            )
            ms = int((time.monotonic() - t0) * 1000)
            data: dict[str, Any] = {
                "rewrites": rewrites,
                "sub_questions": sub_qs,
                "queries": queries,
                "agent_filters": {
                    k: {"values": v, "source": "agent"} for k, v in valid_filters.items()
                },
                "dropped_filters": dropped,
                "rationale": plan_obj.rationale,
                "llm_calls": calls,
            }
            if show_prompts:
                data["prompt"] = user_prompt
            ev = _record(
                "plan",
                "llm",
                f"Planned {len(queries)} queries"
                + (f", {len(valid_filters)} agent filter(s)" if valid_filters else ""),
                ms=ms,
                data=data,
            )
            trace_out = [ev]
            if dropped:
                drop_msg = f"Dropped invalid LLM filter proposal(s): {', '.join(dropped)}"
                trace_out.append(
                    _record("plan_filters", "decision", drop_msg, data={"dropped": dropped})
                )
            return {
                "queries": queries,
                "issued_queries": list(queries),
                "open_questions": sub_qs,
                "agent_filters": valid_filters,
                "llm_calls": state.get("llm_calls", 0) + calls,
                "trace": trace_out,
                "warnings": warnings,
            }
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            msg = f"Agent plan failed ({type(exc).__name__}); continuing with original question."
            warnings.append(msg)
            ev = _record(
                "plan",
                "warning",
                msg,
                ms=ms,
                data={"queries": [q], "error": type(exc).__name__},
            )
            return {
                "queries": [q],
                "issued_queries": [q],
                "open_questions": [],
                "agent_filters": {},
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [ev],
                "warnings": warnings,
            }

    def retrieve_node(state: AgenticState) -> dict[str, Any]:
        q_orig = state["question"].strip()
        queries = state.get("queries") or [q_orig]
        base_spec = state.get("spec") or SearchSpec()
        agent_filters = dict(state.get("agent_filters") or {})
        iteration = state.get("iteration", 0)
        t0 = time.monotonic()
        warnings: list[str] = []
        trace_out: list[TraceEvent] = []

        combined_filters = merge_user_and_agent_filters(base_spec.filters, agent_filters)
        eff_spec = replace(base_spec, filters=combined_filters)

        per_query: dict[str, HybridResponse] = {}
        for q in queries:
            resp = searcher.search(q, eff_spec, top_n=top_n)
            per_query[q] = resp
            warnings.extend(resp.warnings)

        new_passages = merge_and_rerank_passages(q_orig, per_query, searcher, top_n=top_n)

        # Deterministic relaxation rule (ADR-0009 section 6.3 rule 3 & 6.10 item 2):
        # If agent-filtered retrieval returns fewer than min_hits passages, drop agent filters
        # immediately (never user filters) and re-run without spending an LLM call.
        if agent_filters and len(new_passages) < agent_cfg.min_hits:
            relax_msg = (
                f"Agent filters yielded {len(new_passages)} < {agent_cfg.min_hits} passages; "
                "auto-relaxing agent filters (keeping user filters)."
            )
            trace_out.append(
                _record(
                    "relax_filters",
                    "decision",
                    relax_msg,
                    data={
                        "dropped_agent_filters": agent_filters,
                        "user_filters": base_spec.filters,
                    },
                )
            )
            agent_filters = {}
            eff_spec = replace(base_spec, filters=dict(base_spec.filters))
            per_query = {}
            for q in queries:
                resp = searcher.search(q, eff_spec, top_n=top_n)
                per_query[q] = resp
                warnings.extend(resp.warnings)
            new_passages = merge_and_rerank_passages(q_orig, per_query, searcher, top_n=top_n)

        primary = state.get("primary")
        if primary is None:
            primary = per_query.get(queries[0]) or HybridResponse(passages=[], hits=[])

        # Accumulate evidence de-duplicated by (doc_id, position), capped at MAX_EVIDENCE_PASSAGES
        existing = list(state.get("evidence") or [])
        by_key: dict[tuple[str, int], Passage] = {(p.doc_id, p.position): p for p in existing}
        for p in new_passages:
            key = (p.doc_id, p.position)
            if key not in by_key:
                by_key[key] = p
        evidence = list(by_key.values())[:MAX_EVIDENCE_PASSAGES]

        filter_sources = {
            **{k: {"values": list(v), "source": "user"} for k, v in base_spec.filters.items() if v},
            **{k: {"values": list(v), "source": "agent"} for k, v in agent_filters.items() if v},
        }
        ms = int((time.monotonic() - t0) * 1000)
        trace_out.append(
            _record(
                "retrieve",
                "search",
                f"Pass {iteration + 1}: {len(queries)} queries -> {len(evidence)} passages",
                ms=ms,
                data={
                    "iteration": iteration + 1,
                    "queries": queries,
                    "filters": filter_sources,
                    "per_query_legs": {q: r.legs for q, r in per_query.items()},
                    "evidence_count": len(evidence),
                    "evidence": [
                        {"doc_id": p.doc_id, "title": p.title, "path": p.path, "heading": p.heading}
                        for p in evidence
                    ],
                },
            )
        )
        return {
            "evidence": evidence,
            "primary": primary,
            "agent_filters": agent_filters,
            "trace": trace_out,
            "warnings": warnings,
        }

    def grade_node(state: AgenticState) -> dict[str, Any]:
        q = state["question"].strip()
        evidence = state.get("evidence") or []
        open_qs = state.get("open_questions") or []
        issued = state.get("issued_queries") or [q]
        llm_calls = state.get("llm_calls", 0)
        iteration = state.get("iteration", 0)
        started_at = state.get("started_at") or time.monotonic()

        # Check hard budgets before spending an LLM call on grading (reserve 1 call for synthesize)
        elapsed = time.monotonic() - started_at
        if elapsed >= agent_cfg.timeout_s:
            msg = (
                f"Agent timeout budget reached ({elapsed:.1f}s >= {agent_cfg.timeout_s}s); "
                "proceeding to synthesize."
            )
            ev = _record("budget", "warning", msg, data={"budget": "timeout_s"})
            return {
                "grade": {
                    "sufficient": True,
                    "missing": open_qs,
                    "next_queries": [],
                    "budget_hit": "timeout_s",
                },
                "trace": [ev],
                "warnings": [msg],
            }
        if iteration + 1 >= agent_cfg.max_iterations:
            msg = (
                f"Max iterations budget reached ({iteration + 1}/{agent_cfg.max_iterations}); "
                "proceeding to synthesize."
            )
            ev = _record("budget", "warning", msg, data={"budget": "max_iterations"})
            return {
                "grade": {
                    "sufficient": True,
                    "missing": open_qs,
                    "next_queries": [],
                    "budget_hit": "max_iterations",
                },
                "trace": [ev],
                "warnings": [msg],
            }
        if llm_calls + 1 >= agent_cfg.max_llm_calls:
            msg = (
                f"Max LLM calls budget reached ({llm_calls}/{agent_cfg.max_llm_calls}); "
                "proceeding to synthesize."
            )
            ev = _record("budget", "warning", msg, data={"budget": "max_llm_calls"})
            return {
                "grade": {
                    "sufficient": True,
                    "missing": open_qs,
                    "next_queries": [],
                    "budget_hit": "max_llm_calls",
                },
                "trace": [ev],
                "warnings": [msg],
            }
        if len(evidence) >= MAX_EVIDENCE_PASSAGES:
            msg = f"Evidence passage cap reached ({len(evidence)}); proceeding to synthesize."
            ev = _record("budget", "warning", msg, data={"budget": "evidence_cap"})
            return {
                "grade": {
                    "sufficient": True,
                    "missing": open_qs,
                    "next_queries": [],
                    "budget_hit": "evidence_cap",
                },
                "trace": [ev],
                "warnings": [msg],
            }

        user_prompt = build_grade_user_prompt(q, open_qs, evidence, issued)
        t0 = time.monotonic()
        try:
            grade_obj, calls = invoke_structured_with_repair(
                chat_model, Grade, GRADE_SYSTEM_PROMPT, user_prompt
            )
            new_calls = llm_calls + calls
            ms = int((time.monotonic() - t0) * 1000)
            data: dict[str, Any] = {
                "sufficient": grade_obj.sufficient,
                "missing": list(grade_obj.missing),
                "next_queries": list(grade_obj.next_queries)[:3],
                "llm_calls": new_calls,
                "iteration": iteration + 1,
            }
            if show_prompts:
                data["prompt"] = user_prompt
            ev = _record(
                "grade",
                "llm",
                f"Grade: {'sufficient' if grade_obj.sufficient else 'insufficient'} "
                f"({len(grade_obj.missing)} missing)",
                ms=ms,
                data=data,
            )
            trace_out = [ev]
            warnings: list[str] = []
            # If repair retry pushed llm_calls to max_llm_calls, force stop
            sufficient = grade_obj.sufficient
            if not sufficient and new_calls >= agent_cfg.max_llm_calls:
                bmsg = (
                    f"Max LLM calls budget reached ({new_calls}/{agent_cfg.max_llm_calls}); "
                    "proceeding to synthesize."
                )
                trace_out.append(
                    _record("budget", "warning", bmsg, data={"budget": "max_llm_calls"})
                )
                warnings.append(bmsg)
                sufficient = True

            return {
                "grade": {
                    "sufficient": sufficient,
                    "missing": list(grade_obj.missing),
                    "next_queries": list(grade_obj.next_queries)[:3],
                },
                "open_questions": list(grade_obj.missing),
                "llm_calls": new_calls,
                "trace": trace_out,
                "warnings": warnings,
            }
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            msg = f"Evidence grading failed ({type(exc).__name__}); proceeding to synthesize."
            ev = _record("grade", "warning", msg, ms=ms, data={"error": type(exc).__name__})
            return {
                "grade": {"sufficient": True, "missing": open_qs, "next_queries": []},
                "llm_calls": llm_calls + 1,
                "trace": [ev],
                "warnings": [msg],
            }

    def route_after_grade(state: AgenticState) -> str:
        grade = state.get("grade") or {}
        if grade.get("sufficient", True):
            return "synthesize"
        issued_norm = {q.casefold() for q in (state.get("issued_queries") or [])}
        fresh = [
            " ".join(q.split())
            for q in (grade.get("next_queries") or [])
            if " ".join(q.split()) and " ".join(q.split()).casefold() not in issued_norm
        ]
        if not fresh:
            return "synthesize"
        return "refine"

    def refine_node(state: AgenticState) -> dict[str, Any]:
        grade = state.get("grade") or {}
        issued = list(state.get("issued_queries") or [])
        issued_norm = {q.casefold() for q in issued}
        next_qs: list[str] = []
        for q in (grade.get("next_queries") or [])[:3]:
            clean = " ".join(q.split())
            if clean and clean.casefold() not in issued_norm:
                issued_norm.add(clean.casefold())
                next_qs.append(clean)
            if len(next_qs) >= 3:
                break

        new_iter = state.get("iteration", 0) + 1
        ev = _record(
            "refine",
            "decision",
            f"Iteration {new_iter}: refining with {len(next_qs)} new queries",
            data={"iteration": new_iter, "next_queries": next_qs},
        )
        return {
            "queries": next_qs,
            "issued_queries": issued + next_qs,
            "iteration": new_iter,
            "trace": [ev],
        }

    def synthesize_node(state: AgenticState) -> dict[str, Any]:
        evidence = state.get("evidence") or []
        q = state["question"].strip()
        grade = state.get("grade") or {}
        missing = grade.get("missing") or []
        open_qs = state.get("open_questions") or []
        llm_calls = state.get("llm_calls", 0)

        context, sources = build_passages_context(evidence[:top_n])
        if not sources:
            ev = _record(
                "synthesize",
                "decision",
                "No passages retrieved; skipped synthesize LLM call.",
                ms=0,
                data={"sources": 0},
            )
            return {"answer": NO_RESULTS, "sources": [], "trace": [ev]}

        user_prompt = build_answer_user_prompt(q, context, sub_questions=open_qs, missing=missing)
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
                "missing": missing,
                "llm_calls": llm_calls + 1,
                "iterations": state.get("iteration", 0) + 1,
            }
            if show_prompts:
                data["prompt"] = user_prompt
            ev = _record(
                "synthesize",
                "llm",
                f"Synthesized cited answer from {len(sources)} passages",
                ms=ms,
                data=data,
            )
            return {
                "answer": text,
                "sources": sources,
                "llm_calls": llm_calls + 1,
                "trace": [ev],
            }
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            warn = (
                f"Answer synthesis failed ({type(exc).__name__}); "
                "returning retrieved hybrid results."
            )
            ev = _record("synthesize", "warning", warn, ms=ms, data={"error": type(exc).__name__})
            return {
                "answer": "",
                "sources": sources,
                "llm_calls": llm_calls + 1,
                "trace": [ev],
                "warnings": [warn],
            }

    g = StateGraph(AgenticState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("refine", refine_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"synthesize": "synthesize", "refine": "refine"},
    )
    g.add_edge("refine", "retrieve")
    g.add_edge("synthesize", END)
    return g.compile()


def run_agentic_search(
    question: str,
    spec: SearchSpec,
    chat_model: Any,
    searcher: HybridSearcher,
    *,
    cfg: AgentConfig | None = None,
    allowed_facets: Mapping[str, Sequence[str]] | None = None,
    top_n: int = 8,
    sink: TraceSink | None = None,
    show_prompts: bool = False,
) -> AgenticOutput:
    agent_cfg = cfg or AgentConfig()
    graph = build_agentic_graph(
        chat_model,
        searcher,
        cfg=agent_cfg,
        allowed_facets=allowed_facets,
        top_n=top_n,
        sink=sink,
        show_prompts=show_prompts,
    )
    recursion_limit = max(25, agent_cfg.max_iterations * 5 + 10)
    final: AgenticState = graph.invoke(
        {
            "question": question,
            "spec": spec,
            "iteration": 0,
            "llm_calls": 0,
            "started_at": time.monotonic(),
            "evidence": [],
            "trace": [],
            "warnings": [],
        },
        config={"recursion_limit": recursion_limit},
    )
    return AgenticOutput(
        answer=final.get("answer", ""),
        sources=final.get("sources", []),
        primary=final.get("primary") or HybridResponse(passages=[], hits=[]),
        evidence=final.get("evidence", []),
        queries=final.get("issued_queries", [question]),
        trace=final.get("trace", []),
        warnings=final.get("warnings", []),
        llm_calls=final.get("llm_calls", 0),
        iterations=final.get("iteration", 0) + 1,
    )
