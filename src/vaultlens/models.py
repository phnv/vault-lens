"""Plain dataclasses shared by every layer. Stdlib only (see docs/adr/0005)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Literal

MatchMode = Literal["all", "any", "phrase"]
SortKey = Literal["relevance", "modified_desc", "created_desc", "title_asc"]


@dataclass(frozen=True)
class Chunk:
    position: int
    heading: str
    level: int
    text: str


@dataclass
class Note:
    vault: str
    path: str  # posix, relative to vault root
    doc_id: str
    title: str
    body: str
    content_hash: str
    folder: str
    note_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    type: str | None = None
    status: str | None = None
    created: str | None = None
    modified: str | None = None
    links_out: list[str] = field(default_factory=list)  # lowercased, unresolved names/paths
    urls: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    size_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_es_doc(self) -> dict[str, Any]:
        """Document body matching search/mappings/notes_v1.json (dynamic: strict)."""
        return {
            "vault": self.vault,
            "path": self.path,
            "note_id": self.note_id,
            "title": self.title,
            "aliases": self.aliases,
            "body": self.body,
            "type": self.type,
            "status": self.status,
            "tags": self.tags,
            "created": self.created,
            "modified": self.modified,
            "folder": self.folder,
            "links_out": self.links_out,
            "urls": self.urls,
            "meta": self.meta,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunks": [asdict(c) for c in self.chunks],
        }


@dataclass
class SearchSpec:
    """Every UI control maps to one field here (docs/SPEC.md). JSON-serialisable."""

    query: str = ""
    vaults: list[str] = field(default_factory=list)
    match_mode: MatchMode = "all"
    match_type: str = "best_fields"
    fields: dict[str, float] = field(
        default_factory=lambda: {"title": 3.0, "aliases": 2.0, "body.pt": 1.0, "body.en": 1.0}
    )
    fuzziness: str | None = None
    minimum_should_match: str | None = None
    filters: dict[str, list[str]] = field(default_factory=dict)  # OR within a field, AND across
    date_range: dict[str, str] | None = None  # {"field": "modified", "gte": "...", "lte": "..."}
    sort: SortKey = "relevance"
    size: int = 20
    from_: int = 0
    highlight: bool = True
    facets: bool = True
    with_chunks: bool = True  # nested chunk inner_hits ("best chunks"); off for rank_eval
    chunk_text: bool = False  # return chunk text (needed by Ask)
    chunk_size: int = 3  # inner_hits per note
    chunk_boost: float = 0.2
    explain: bool = False
    profile: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchSpec:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ChunkHit:
    heading: str
    position: int
    text: str = ""
    snippet: str = ""  # html-escaped, <mark> tags


@dataclass
class Hit:
    doc_id: str
    score: float | None
    title: str
    path: str
    vault: str
    folder: str = ""
    type: str | None = None
    status: str | None = None
    tags: list[str] = field(default_factory=list)
    modified: str | None = None
    created: str | None = None
    indexed_at: str | None = None
    note_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    links_out: list[str] = field(default_factory=list)
    snippet: str = ""  # html-escaped, <mark> tags
    chunks: list[ChunkHit] = field(default_factory=list)
    explanation: dict[str, Any] | None = None


@dataclass
class SearchResponse:
    total: int
    took_ms: int
    hits: list[Hit]
    facets: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
