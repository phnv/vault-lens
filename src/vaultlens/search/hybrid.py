"""Client-side Reciprocal Rank Fusion and hybrid lexical + semantic search (ADR-0009 section 6.6).

Pure fusion helpers plus `HybridSearcher` orchestrator; no `elasticsearch` import.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ..config import HybridConfig, RerankConfig
from ..models import ChunkHit, Hit, HybridResponse, Passage, SearchSpec, TraceEvent
from ..semantic.embedder import Embedder
from ..semantic.reranker import Reranker, prepare_rerank_text
from .query_builder import build_search_body, filter_clauses
from .results import parse_response, strip_marks

TraceCallback = Callable[[TraceEvent], None]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def rrf_fuse(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over ordered item-id lists (best first).

    Returns `[(item_id, score)]` sorted desc, tie-breaking deterministically on `item_id`.
    """
    if not rankings:
        return []
    ws = weights if weights is not None else [1.0] * len(rankings)
    if len(ws) != len(rankings):
        raise ValueError("weights must match rankings length")
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, ws, strict=True):
        seen: set[str] = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            scores[item] = scores.get(item, 0.0) + (w / (k + rank))
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def knn_body(
    vector: list[float],
    spec: SearchSpec,
    k: int,
    num_candidates: int,
    inner_size: int = 2,
) -> dict[str, Any]:
    """Nested kNN request body for `chunks.embedding` with parent-level filters and `inner_hits`."""
    filters = filter_clauses(spec)
    knn: dict[str, Any] = {
        "field": "chunks.embedding",
        "query_vector": vector,
        "k": k,
        "num_candidates": num_candidates,
        "inner_hits": {
            "size": inner_size,
            "_source": ["chunks.heading", "chunks.position", "chunks.text"],
        },
    }
    if filters:
        knn["filter"] = {"bool": {"filter": filters}} if len(filters) > 1 else filters[0]
    return {
        "knn": knn,
        "size": k,
        "_source": {"excludes": ["body", "chunks"]},
    }


class HybridSearcher:
    def __init__(
        self,
        search_fn: Callable[[dict[str, Any]], dict[str, Any]],
        embedder: Embedder,
        reranker: Reranker | None = None,
        cfg: HybridConfig | None = None,
        rerank_cfg: RerankConfig | None = None,
        trace: TraceCallback | None = None,
        oversized_chars: int = 1600,
    ) -> None:
        self.search_fn = search_fn
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg or HybridConfig()
        self.rerank_cfg = rerank_cfg or RerankConfig()
        self.trace = trace
        self.oversized_chars = oversized_chars

    def _emit(
        self,
        step: str,
        kind: str,
        summary: str,
        ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if self.trace is not None:
            self.trace(
                TraceEvent(
                    ts=_now_iso(),
                    step=step,
                    kind=kind,
                    summary=summary,
                    ms=ms,
                    data=dict(data or {}),
                )
            )

    def search(
        self,
        query: str,
        spec: SearchSpec | None = None,
        *,
        top_n: int = 10,
        rerank: bool | None = None,
    ) -> HybridResponse:
        base_spec = replace(spec or SearchSpec(), query=query)
        q_clean = query.strip()
        warnings: list[str] = []
        t_start = time.monotonic()

        # Empty query falls back to lexical match_all behaviour, skips kNN and rerank
        if not q_clean:
            lex_body = build_search_body(base_spec)
            t0 = time.monotonic()
            raw = self.search_fn(lex_body)
            lex_resp = parse_response(raw)
            ms = int((time.monotonic() - t0) * 1000)
            self._emit(
                "lexical",
                "search",
                f"empty query match_all -> {len(lex_resp.hits)} hits",
                ms=ms,
                data={"lexical_hits": len(lex_resp.hits)},
            )
            return HybridResponse(
                passages=[],
                hits=lex_resp.hits,
                facets=lex_resp.facets,
                total=lex_resp.total,
                took_ms=lex_resp.took_ms or ms,
                legs={"lexical": len(lex_resp.hits), "semantic": 0},
                warnings=warnings,
            )

        depth = max(self.cfg.candidates, base_spec.size, top_n)
        lex_spec = replace(
            base_spec,
            query=q_clean,
            with_chunks=True,
            chunk_text=True,
            size=depth,
        )

        lex_resp = None
        sem_resp = None
        lex_err: Exception | None = None
        sem_err: Exception | None = None

        # 1. Lexical leg
        t_lex = time.monotonic()
        try:
            lex_raw = self.search_fn(build_search_body(lex_spec))
            lex_resp = parse_response(lex_raw)
            self._emit(
                "lexical",
                "search",
                f"BM25 retrieved {len(lex_resp.hits)} notes",
                ms=int((time.monotonic() - t_lex) * 1000),
                data={"query": q_clean, "hits": len(lex_resp.hits)},
            )
        except Exception as exc:
            lex_err = exc
            msg = f"Lexical leg failed ({type(exc).__name__}); continuing with semantic leg."
            warnings.append(msg)
            self._emit("lexical", "warning", msg, data={"error": type(exc).__name__})

        # 2. Semantic leg
        t_sem = time.monotonic()
        try:
            vec = self.embedder.embed_query(q_clean)
            k_body = knn_body(
                vec,
                lex_spec,
                k=depth,
                num_candidates=depth * 3,
                inner_size=lex_spec.chunk_size,
            )
            sem_raw = self.search_fn(k_body)
            sem_resp = parse_response(sem_raw)
            self._emit(
                "semantic",
                "search",
                f"kNN retrieved {len(sem_resp.hits)} notes",
                ms=int((time.monotonic() - t_sem) * 1000),
                data={"query": q_clean, "hits": len(sem_resp.hits)},
            )
        except Exception as exc:
            sem_err = exc
            msg = f"Semantic leg failed ({type(exc).__name__}); continuing with lexical leg."
            warnings.append(msg)
            self._emit("semantic", "warning", msg, data={"error": type(exc).__name__})

        if lex_resp is None and sem_resp is None:
            raise RuntimeError(
                f"Both hybrid legs failed: lexical={type(lex_err).__name__}, "
                f"semantic={type(sem_err).__name__}"
            ) from (lex_err or sem_err)

        lex_hits = lex_resp.hits if lex_resp else []
        sem_hits = sem_resp.hits if sem_resp else []

        # 4. Fuse at note level
        lex_ids = [h.doc_id for h in lex_hits]
        sem_ids = [h.doc_id for h in sem_hits]
        rankings = [r for r in (lex_ids, sem_ids) if r]
        fused_notes = rrf_fuse(rankings, k=self.cfg.rrf_k)
        note_rrf = dict(fused_notes)

        hits_by_id: dict[str, Hit] = {}
        for h in sem_hits:
            hits_by_id[h.doc_id] = h
        for h in lex_hits:
            # Prefer lexical hit metadata/snippet when available, but merge chunks below
            hits_by_id[h.doc_id] = h

        lex_chunks_by_id: dict[str, list[ChunkHit]] = {h.doc_id: list(h.chunks) for h in lex_hits}
        sem_chunks_by_id: dict[str, list[ChunkHit]] = {h.doc_id: list(h.chunks) for h in sem_hits}

        # 5. Passage candidates from top RERANK_POOL fused notes
        pool_note_ids = [doc_id for doc_id, _ in fused_notes[: self.rerank_cfg.pool]]
        passages_by_key: dict[str, Passage] = {}
        snippets_by_key: dict[str, str] = {}
        initial_passage_order: list[str] = []

        for doc_id in pool_note_ids:
            hit = hits_by_id[doc_id]
            combined_chunks = lex_chunks_by_id.get(doc_id, []) + sem_chunks_by_id.get(doc_id, [])
            added_for_note = False
            for ch in combined_chunks:
                if not ch.text:
                    continue
                added_for_note = True
                pkey = f"{doc_id}#{ch.position}"
                if pkey not in passages_by_key:
                    passages_by_key[pkey] = Passage(
                        doc_id=doc_id,
                        path=hit.path,
                        title=hit.title,
                        heading=ch.heading,
                        vault=hit.vault,
                        position=ch.position,
                        text=ch.text,
                        rrf_score=note_rrf.get(doc_id),
                    )
                    snippets_by_key[pkey] = ch.snippet
                    initial_passage_order.append(pkey)
                elif ch.snippet and not snippets_by_key.get(pkey):
                    snippets_by_key[pkey] = ch.snippet
            if not added_for_note:
                fallback_text = (strip_marks(hit.snippet).strip() if hit.snippet else "") or (
                    hit.title or hit.path
                )
                if fallback_text:
                    pkey = f"{doc_id}#-1"
                    passages_by_key[pkey] = Passage(
                        doc_id=doc_id,
                        path=hit.path,
                        title=hit.title,
                        heading="",
                        vault=hit.vault,
                        position=-1,
                        text=fallback_text,
                        rrf_score=note_rrf.get(doc_id),
                    )
                    snippets_by_key[pkey] = hit.snippet or ""
                    initial_passage_order.append(pkey)

        # 6. Rerank passages (if enabled and reranker is available)
        use_rerank = (
            self.rerank_cfg.enabled if rerank is None else rerank
        ) and self.reranker is not None
        final_passage_keys = list(initial_passage_order)

        if use_rerank and initial_passage_order and self.reranker is not None:
            t_rr = time.monotonic()
            try:
                texts_to_score = [
                    prepare_rerank_text(
                        q_clean,
                        passages_by_key[pk].heading,
                        passages_by_key[pk].text,
                        snippets_by_key.get(pk, ""),
                        oversized_chars=self.oversized_chars,
                    )
                    for pk in initial_passage_order
                ]
                scores = self.reranker.score_pairs(q_clean, texts_to_score)
                for pk, sc in zip(initial_passage_order, scores, strict=True):
                    passages_by_key[pk].rerank_score = float(sc)

                reranker_order = [
                    pk
                    for pk, _ in sorted(
                        zip(initial_passage_order, scores, strict=True),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                ]
                if self.rerank_cfg.mode == "replace":
                    final_passage_keys = reranker_order
                else:
                    blended = rrf_fuse([initial_passage_order, reranker_order], k=self.cfg.rrf_k)
                    for pk, bscore in blended:
                        passages_by_key[pk].rrf_score = bscore
                    final_passage_keys = [pk for pk, _ in blended]

                self._emit(
                    "rerank",
                    "rerank",
                    f"Reranked {len(initial_passage_order)} passages ({self.rerank_cfg.mode})",
                    ms=int((time.monotonic() - t_rr) * 1000),
                    data={
                        "mode": self.rerank_cfg.mode,
                        "passages": len(initial_passage_order),
                    },
                )
            except Exception as exc:
                msg = f"Reranker failed ({type(exc).__name__}); using RRF order."
                warnings.append(msg)
                self._emit("rerank", "warning", msg, data={"error": type(exc).__name__})

        ordered_passages = [passages_by_key[pk] for pk in final_passage_keys]
        top_passages = ordered_passages[:top_n]

        # 7. Build note-level hits ordered by best passage, then remaining fused notes
        note_order: list[str] = []
        seen_notes: set[str] = set()
        chunks_for_note: dict[str, list[ChunkHit]] = {}

        for pk in final_passage_keys:
            p = passages_by_key[pk]
            if p.doc_id not in seen_notes:
                seen_notes.add(p.doc_id)
                note_order.append(p.doc_id)
            chunks_for_note.setdefault(p.doc_id, []).append(
                ChunkHit(
                    heading=p.heading,
                    position=p.position,
                    text=p.text,
                    snippet=snippets_by_key.get(pk, ""),
                )
            )
        for doc_id, _ in fused_notes:
            if doc_id not in seen_notes:
                seen_notes.add(doc_id)
                note_order.append(doc_id)

        final_hits: list[Hit] = []
        for doc_id in note_order[: base_spec.size]:
            base_hit = hits_by_id[doc_id]
            attached = chunks_for_note.get(doc_id) or base_hit.chunks
            best_score = chunks_for_note[doc_id][0] if doc_id in chunks_for_note else None
            score_val = note_rrf.get(doc_id)
            if best_score is not None:
                first_pk = f"{doc_id}#{best_score.position}"
                p_obj = passages_by_key.get(first_pk)
                if p_obj is not None:
                    score_val = (
                        p_obj.rerank_score
                        if (
                            use_rerank
                            and self.rerank_cfg.mode == "replace"
                            and p_obj.rerank_score is not None
                        )
                        else p_obj.rrf_score
                    )
            final_hits.append(
                replace(
                    base_hit,
                    score=round(float(score_val), 6) if score_val is not None else base_hit.score,
                    chunks=attached[: base_spec.chunk_size],
                )
            )

        facets = lex_resp.facets if lex_resp else (sem_resp.facets if sem_resp else {})
        total = max(
            lex_resp.total if lex_resp else 0,
            sem_resp.total if sem_resp else 0,
            len(fused_notes),
        )
        took_ms = int((time.monotonic() - t_start) * 1000)
        return HybridResponse(
            passages=top_passages,
            hits=final_hits,
            facets=facets,
            total=total,
            took_ms=took_ms,
            legs={"lexical": len(lex_hits), "semantic": len(sem_hits)},
            warnings=warnings,
        )
