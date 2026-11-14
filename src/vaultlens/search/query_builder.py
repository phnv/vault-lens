"""SearchSpec -> Elasticsearch request body. Pure, deterministic, no I/O.

The Workbench "compiled query" view shows exactly what this returns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from typing import Any

from ..models import SearchSpec

ALLOWED_FILTER_FIELDS = {"vault", "type", "status", "tags", "folder", "folder.tree", "links_out"}
ALLOWED_QUERY_FIELDS = {"title", "title.edge", "aliases", "body", "body.pt", "body.en"}
ALLOWED_MATCH_TYPES = {
    "best_fields",
    "most_fields",
    "cross_fields",
    "phrase",
    "phrase_prefix",
    "bool_prefix",
}
FUZZY_TYPES = {"best_fields", "most_fields", "bool_prefix"}
PHRASE_TYPES = {"phrase", "phrase_prefix"}
DATE_FIELDS = {"created", "modified"}
_META_FIELD = re.compile(r"^meta\.[\w.\-]+$")

SORTS: dict[str, list[Any] | None] = {
    "relevance": None,
    "modified_desc": [{"modified": {"order": "desc", "missing": "_last"}}, "_score"],
    "created_desc": [{"created": {"order": "desc", "missing": "_last"}}, "_score"],
    "title_asc": [{"title.raw": {"order": "asc", "missing": "_last"}}, "_score"],
}

# Strategies compared on the Evaluate page (docs/SPEC.md > Evaluate). Add columns here.
STRATEGIES: dict[str, dict[str, Any]] = {
    "A": {"label": "plain BM25", "fields": {"body": 1.0}},
    "B": {"label": "boosted fields", "fields": {"title": 3.0, "aliases": 2.0, "body": 1.0}},
    "C": {
        "label": "pt/en analyzers",
        "fields": {"title": 3.0, "aliases": 2.0, "body.pt": 1.0, "body.en": 1.0},
    },
}


META_PREFIX = "meta."


def date_range_spec(field_name: str, dates: Sequence[date]) -> dict[str, str] | None:
    """Picked dates (none, start only, or start+end) -> SearchSpec.date_range, else None."""
    if field_name not in DATE_FIELDS:
        raise ValueError(f"date field not allowed: {field_name!r}")
    if not dates:
        return None
    out = {"field": field_name, "gte": dates[0].isoformat()}
    if len(dates) > 1:
        out["lte"] = dates[1].isoformat()
    return out


def meta_filters(rows: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    """UI rows of (custom key, values) -> SearchSpec.filters entries.

    Blank keys or empty values are skipped, invalid keys are ignored (never reach Elasticsearch),
    and repeated keys are merged (OR within a key).
    """
    out: dict[str, list[str]] = {}
    for key, values in rows:
        name = META_PREFIX + key.strip().removeprefix(META_PREFIX)
        if not key.strip() or not values or not _META_FIELD.match(name):
            continue
        merged = out.setdefault(name, [])
        merged.extend(v for v in values if v not in merged)
    return out


def meta_rows(filters: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """Inverse of meta_filters, e.g. to restore a saved search into the UI."""
    return [
        (name.removeprefix(META_PREFIX), list(vals))
        for name, vals in sorted(filters.items())
        if name.startswith(META_PREFIX) and vals
    ]


def meta_values_body(key: str, vault: str | None = None, size: int = 50) -> dict[str, Any]:
    """Distinct values of one custom field (flattened `meta.<key>`), with doc counts."""
    name = META_PREFIX + key.strip().removeprefix(META_PREFIX)
    if not _META_FIELD.match(name):
        raise ValueError(f"filter field not allowed: {name!r}")
    flt = [{"term": {"vault": vault}}] if vault else []
    return {
        "size": 0,
        "query": {"bool": {"filter": flt}},
        "aggs": {"values": {"terms": {"field": name, "size": size}}},
    }


def _check_filter_field(name: str) -> None:
    if name not in ALLOWED_FILTER_FIELDS and not _META_FIELD.match(name):
        raise ValueError(f"filter field not allowed: {name!r}")


def _field_list(fields: dict[str, float]) -> list[str]:
    out = []
    for name, boost in fields.items():
        if name not in ALLOWED_QUERY_FIELDS:
            raise ValueError(f"query field not allowed: {name!r}")
        out.append(name if boost == 1 else f"{name}^{boost:g}")
    return out


def _operator(spec: SearchSpec) -> str:
    return "and" if spec.match_mode == "all" else "or"


def _text_clause(spec: SearchSpec) -> dict[str, Any]:
    q = spec.query.strip()
    if not q:
        return {"match_all": {}}
    if spec.match_type not in ALLOWED_MATCH_TYPES:
        raise ValueError(f"match_type not allowed: {spec.match_type!r}")
    mtype = "phrase" if spec.match_mode == "phrase" else spec.match_type
    mm: dict[str, Any] = {"query": q, "type": mtype, "fields": _field_list(spec.fields)}
    if mtype not in PHRASE_TYPES:
        mm["operator"] = _operator(spec)
        if mm["operator"] == "or" and spec.minimum_should_match:
            mm["minimum_should_match"] = spec.minimum_should_match
        if spec.fuzziness and mtype in FUZZY_TYPES:
            mm["fuzziness"] = spec.fuzziness
    return {"multi_match": mm}


def _chunk_clause(spec: SearchSpec) -> dict[str, Any]:
    """Nested query used for 'best chunks' (inner_hits). Scores lightly via chunk_boost."""
    q = spec.query.strip()
    mm: dict[str, Any] = {
        "query": q,
        "fields": ["chunks.text.pt", "chunks.text.en", "chunks.heading^2"],
        "type": "phrase" if spec.match_mode == "phrase" else "best_fields",
    }
    if mm["type"] != "phrase":
        mm["operator"] = _operator(spec)
        if mm["operator"] == "or" and spec.minimum_should_match:
            mm["minimum_should_match"] = spec.minimum_should_match
    src = ["chunks.heading", "chunks.position"] + (["chunks.text"] if spec.chunk_text else [])
    return {
        "nested": {
            "path": "chunks",
            "score_mode": "max",
            "boost": spec.chunk_boost,
            "query": {"multi_match": mm},
            "inner_hits": {
                "size": spec.chunk_size,
                "_source": src,
                "highlight": {
                    "encoder": "html",
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "fields": {"chunks.text.pt": {"number_of_fragments": 1, "fragment_size": 160}},
                },
            },
        }
    }


def filter_clauses(spec: SearchSpec) -> list[dict[str, Any]]:
    """Public filter clauses for `spec` (reused by lexical search and nested kNN)."""
    return _filters(spec)


def _filters(spec: SearchSpec) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    if spec.vaults:
        clauses.append({"terms": {"vault": spec.vaults}})
    for field_name, values in sorted(spec.filters.items()):
        if not values:
            continue
        _check_filter_field(field_name)
        clauses.append({"terms": {field_name: list(values)}})
    if spec.date_range:
        dr = spec.date_range
        field_name = dr.get("field", "modified")
        if field_name not in DATE_FIELDS:
            raise ValueError(f"date field not allowed: {field_name!r}")
        bounds = {k: dr[k] for k in ("gte", "lte") if dr.get(k)}
        if bounds:
            clauses.append({"range": {field_name: bounds}})
    return clauses


def _highlight(spec: SearchSpec) -> dict[str, Any]:
    names = [f for f in spec.fields if f.startswith("body")] or ["title"]
    return {
        "encoder": "html",
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fields": {n: {"number_of_fragments": 1, "fragment_size": 160} for n in names},
    }


def build_search_body(spec: SearchSpec) -> dict[str, Any]:
    bool_q: dict[str, Any] = {"must": [_text_clause(spec)], "filter": _filters(spec)}
    if spec.query.strip() and spec.with_chunks:
        bool_q["should"] = [_chunk_clause(spec)]
    body: dict[str, Any] = {
        "query": {"bool": bool_q},
        "size": spec.size,
        "from": spec.from_,
        "track_total_hits": True,
        "_source": {"excludes": ["body", "chunks"]},
    }
    sort = SORTS.get(spec.sort)
    if sort:
        body["sort"] = sort
    if spec.highlight and spec.query.strip():
        body["highlight"] = _highlight(spec)
    if spec.facets:
        body["aggs"] = {
            "type": {"terms": {"field": "type", "size": 10}},
            "status": {"terms": {"field": "status", "size": 10}},
            "folder": {"terms": {"field": "folder.tree", "size": 50}},
            "tags": {"terms": {"field": "tags", "size": 30}},
        }
    if spec.explain:
        body["explain"] = True
    if spec.profile:
        body["profile"] = True
    return body


def compiled_query(spec: SearchSpec) -> str:
    return json.dumps(build_search_body(spec), indent=2, ensure_ascii=False)


def strategy_spec(
    name: str, query: str, vaults: list[str] | None = None, size: int = 10
) -> SearchSpec:
    """Spec for Evaluate: no chunks/facets/highlight so only note-level ranking counts."""
    cfg = STRATEGIES[name]
    return replace(
        SearchSpec(),
        query=query,
        vaults=list(vaults or []),
        fields=dict(cfg["fields"]),
        match_mode="any",
        size=size,
        with_chunks=False,
        facets=False,
        highlight=False,
    )


def inbound_links_body(
    vault: str, path: str, title: str, aliases: list[str], size: int = 50
) -> dict[str, Any]:
    """'In: N' count and list: notes whose links_out points at this note (name, alias or path)."""
    stem = path.rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.lower().endswith(".md") else stem
    no_ext = path[:-3] if path.lower().endswith(".md") else path
    targets = sorted({t.lower() for t in [stem, title, no_ext, *aliases] if t})
    return {
        "query": {
            "bool": {
                "filter": [{"term": {"vault": vault}}, {"terms": {"links_out": targets}}],
                "must_not": [{"term": {"path": path}}],
            }
        },
        "size": size,
        "_source": ["title", "path"],
    }


def more_like_this_body(doc_id: str, vault: str | None = None, size: int = 5) -> dict[str, Any]:
    flt = [{"term": {"vault": vault}}] if vault else []
    return {
        "query": {
            "bool": {
                "must": [
                    {
                        "more_like_this": {
                            "fields": ["title", "body.pt", "body.en"],
                            "like": [{"_id": doc_id}],
                            "min_term_freq": 1,
                            "min_doc_freq": 1,
                        }
                    }
                ],
                "filter": flt,
            }
        },
        "size": size,
        "_source": {"excludes": ["body", "chunks"]},
    }
