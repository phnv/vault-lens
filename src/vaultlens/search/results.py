"""Raw Elasticsearch response -> SearchResponse. Pure, so it is unit-testable offline."""
from __future__ import annotations

import html
import re
from typing import Any

from ..models import ChunkHit, Hit, SearchResponse

_MARK = re.compile(r"</?mark>")


def strip_marks(snippet: str) -> str:
    return html.unescape(_MARK.sub("", snippet))


def _total(raw: dict[str, Any]) -> int:
    total = raw.get("hits", {}).get("total", 0)
    return int(total["value"]) if isinstance(total, dict) else int(total or 0)


def _chunk_hits(hit: dict[str, Any]) -> list[ChunkHit]:
    inner = hit.get("inner_hits", {}).get("chunks", {}).get("hits", {}).get("hits", [])
    out = []
    for ih in inner:
        # nested inner_hits _source is the nested object itself, possibly wrapped under "chunks"
        src = ih.get("_source", {})
        src = src.get("chunks", src)
        frag = (ih.get("highlight", {}) or {}).get("chunks.text.pt", [""])
        out.append(ChunkHit(
            heading=src.get("heading", ""),
            position=int(src.get("position", ih.get("_nested", {}).get("offset", 0))),
            text=src.get("text", ""),
            snippet=frag[0] if frag else "",
        ))
    return out


def parse_hit(h: dict[str, Any]) -> Hit:
    src = h.get("_source", {})
    hl = h.get("highlight", {}) or {}
    snippet = next((v[0] for v in hl.values() if v), "")
    return Hit(
        doc_id=h["_id"],
        score=h.get("_score"),
        title=src.get("title", ""),
        path=src.get("path", ""),
        vault=src.get("vault", ""),
        folder=src.get("folder", ""),
        type=src.get("type"),
        status=src.get("status"),
        tags=src.get("tags", []) or [],
        modified=src.get("modified"),
        created=src.get("created"),
        indexed_at=src.get("indexed_at"),
        note_id=src.get("note_id"),
        aliases=src.get("aliases", []) or [],
        links_out=src.get("links_out", []) or [],
        snippet=snippet,
        chunks=_chunk_hits(h),
        explanation=h.get("_explanation"),
    )


def parse_response(raw: dict[str, Any]) -> SearchResponse:
    facets = {
        name: [(b["key"], b["doc_count"]) for b in agg.get("buckets", [])]
        for name, agg in raw.get("aggregations", {}).items()
    }
    return SearchResponse(
        total=_total(raw),
        took_ms=int(raw.get("took", 0)),
        hits=[parse_hit(h) for h in raw.get("hits", {}).get("hits", [])],
        facets=facets,
        raw=raw,
    )
