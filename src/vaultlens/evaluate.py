"""Judged-query evaluation via the Elasticsearch _rank_eval API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .search.query_builder import STRATEGIES, build_search_body, strategy_spec


@dataclass(frozen=True)
class JudgedQuery:
    id: str
    query: str
    vaults: list[str]
    ratings: dict[str, int]  # doc_id -> graded relevance 0..3


@dataclass
class StrategyResult:
    strategy: str
    label: str
    mean: float
    per_query: dict[str, float] = field(default_factory=dict)


def load_judgments(path: str | Path) -> list[JudgedQuery]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: list[JudgedQuery] = []
    for item in raw.get("queries") or []:
        ratings: dict[str, int] = {}
        for rel in item.get("relevant") or []:
            rating = int(rel.get("rating", 1))
            if not 0 <= rating <= 3:
                raise ValueError(f"rating must be 0..3 in query {item.get('id')!r}")
            ratings[str(rel["doc_id"])] = rating
        if not item.get("query") or not ratings:
            raise ValueError(f"query {item.get('id')!r} needs text and at least one judgment")
        out.append(
            JudgedQuery(str(item["id"]), item["query"], list(item.get("vaults") or []), ratings)
        )
    return out


def build_rank_eval_body(
    judged: list[JudgedQuery], strategy: str, concrete_index: str, k: int = 10
) -> dict[str, Any]:
    """rank_eval matches hits by concrete index name, not by alias."""
    requests = []
    for q in judged:
        body = build_search_body(strategy_spec(strategy, q.query, q.vaults, size=k))
        requests.append(
            {
                "id": q.id,
                "request": {"query": body["query"]},
                "ratings": [
                    {"_index": concrete_index, "_id": doc_id, "rating": r}
                    for doc_id, r in sorted(q.ratings.items())
                ],
            }
        )
    return {"requests": requests, "metric": {"ndcg": {"k": k, "normalize": True}}}


def parse_rank_eval(strategy: str, response: dict[str, Any]) -> StrategyResult:
    per_query = {qid: d["metric_score"] for qid, d in response.get("details", {}).items()}
    return StrategyResult(
        strategy=strategy,
        label=STRATEGIES[strategy]["label"],
        mean=float(response.get("metric_score", 0.0)),
        per_query=per_query,
    )


def run_evaluation(backend: Any, judged: list[JudgedQuery], strategies: list[str], k: int = 10):
    index = backend.current_index()
    if index is None:
        raise RuntimeError("no index yet; run a sync first")
    return [
        parse_rank_eval(s, backend.rank_eval(build_rank_eval_body(judged, s, index, k)))
        for s in strategies
    ]
