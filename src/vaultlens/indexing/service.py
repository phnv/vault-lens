"""Vault -> index synchronisation. Incremental (hash diff) and full (build vN+1, swap alias)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..config import Settings, VaultConfig
from ..models import Note
from ..search.backend import SearchBackend
from ..semantic.embedder import Embedder, EmbeddingError, note_embedding_inputs
from ..store import JsonStore
from ..vault.parser import get_chunker, get_parser
from ..vault.scanner import diff_hashes, scan_vault


class IndexCompatibilityError(RuntimeError):
    """Raised when the live index's _meta is incompatible and requires a full reindex."""


@dataclass
class SyncReport:
    vault: str
    scanned: int = 0
    new: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: int = 0
    seconds: float = 0.0
    embedded: int = 0
    embed_seconds: float = 0.0
    index: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_duplicate_ids(notes: list[Note]) -> None:
    """Two notes sharing a frontmatter id fall back to a path-based doc id on the later path."""
    seen: set[str] = set()
    for note in sorted(notes, key=lambda n: n.path):
        if note.doc_id in seen:
            note.warnings.append(
                f"duplicate frontmatter id {note.note_id!r}; using path-based doc id"
            )
            note.doc_id = f"{note.vault}:path:{note.path}"
        seen.add(note.doc_id)


class IndexService:
    def __init__(
        self,
        backend: SearchBackend,
        settings: Settings,
        store: JsonStore,
        embedder: Embedder | None = None,
    ) -> None:
        self.backend, self.settings, self.store = backend, settings, store
        self.embedder = embedder
        self._embedded = 0
        self._embed_seconds = 0.0

    def load_notes(self, vault: VaultConfig) -> tuple[list[Note], list[str]]:
        parser_fn = get_parser(self.settings.parser)
        get_chunker(self.settings.chunker)  # validate chunker name upfront
        notes: list[Note] = []
        errors: list[str] = []
        for raw in scan_vault(vault):
            try:
                notes.append(
                    parser_fn(
                        vault.slug,
                        raw.path,
                        raw.text,
                        raw.mtime,
                        self.settings.chunk_max_chars,
                        chunker=self.settings.chunker,
                        chunk_overlap=self.settings.chunk_overlap,
                        parser=self.settings.parser,
                        embedding_model=self.settings.embedding_model,
                    )
                )
            except Exception as exc:  # one bad file must not abort a sync
                errors.append(f"{raw.path}: {type(exc).__name__}")
        resolve_duplicate_ids(notes)
        return notes, errors

    def _embed_note(self, note: Note, errors: list[str]) -> list[list[float]] | None:
        """Vectors for one note, or None (embeddings off, no chunks, or a recorded failure)."""
        if self.embedder is None or not self.settings.embeddings_enabled or not note.chunks:
            return None
        t0 = time.monotonic()
        try:
            vectors = self.embedder.embed_documents(note_embedding_inputs(note))
            if len(vectors) != len(note.chunks):
                raise ValueError("embedder returned a different number of vectors")
            bad = [v for v in vectors if len(v) != self.settings.embed.dims]
            if bad:
                raise ValueError(f"vector dims != EMBED_DIMS ({self.settings.embed.dims})")
        except Exception as exc:  # one note must not abort a sync; it stays BM25-only
            errors.append(f"{note.path}: embedding failed ({type(exc).__name__})")
            return None
        self._embedded += 1
        self._embed_seconds += time.monotonic() - t0
        return vectors

    def _index_action(
        self, index: str, note: Note, errors: list[str] | None = None
    ) -> dict[str, Any]:
        vectors = self._embed_note(note, errors if errors is not None else [])
        return {
            "_op_type": "index",
            "_index": index,
            "_id": note.doc_id,
            "_source": note.to_es_doc(vectors),
        }

    def _require_embedder(self) -> None:
        if self.settings.embeddings_enabled and self.embedder is None:
            raise EmbeddingError(
                f"Embeddings are enabled (embedding_model={self.settings.embedding_model!r}) "
                "but no embedder was configured."
            )

    def sync(self, vault_slug: str, force: bool = False) -> SyncReport:
        """Incremental: index new/changed notes, delete removed ones."""
        t0 = time.monotonic()
        self._embedded, self._embed_seconds = 0, 0.0
        vault = self.settings.vault(vault_slug)
        ok, reason = self.backend.check_index_compatibility(self.settings.embedding_model)
        if not ok:
            raise IndexCompatibilityError(reason)
        self._require_embedder()
        index = self.backend.ensure_index(self.settings.embedding_model)
        notes, errors = self.load_notes(vault)
        by_id = {n.doc_id: n for n in notes}
        d = diff_hashes(
            {i: n.content_hash for i, n in by_id.items()},
            self.backend.indexed_hashes(vault.slug),
            force=force,
        )
        actions = [
            self._index_action(self.backend.alias, by_id[i], errors) for i in d.new + d.changed
        ]
        actions += [
            {"_op_type": "delete", "_index": self.backend.alias, "_id": i} for i in d.removed
        ]
        if actions:
            _, bulk_errors = self.backend.bulk(actions)
            errors += [str(e)[:200] for e in bulk_errors]
            self.backend.refresh()
        return self._finish(
            vault.slug, index, notes, d.new, d.changed, d.removed, d.unchanged, errors, t0
        )

    def full_reindex(self) -> list[SyncReport]:
        """Build notes_vN+1 from every registered vault, then swap the alias atomically."""
        self._require_embedder()
        old = self.backend.current_index()
        new = self.backend.create_next_index(self.settings.embedding_model)
        reports: list[SyncReport] = []
        total_with_chunks = 0
        total_embedded = 0
        for vault in self.settings.vaults:
            t0 = time.monotonic()
            self._embedded, self._embed_seconds = 0, 0.0
            notes, errors = self.load_notes(vault)
            total_with_chunks += sum(1 for n in notes if n.chunks)
            _, bulk_errors = self.backend.bulk(self._index_action(new, n, errors) for n in notes)
            errors += [str(e)[:200] for e in bulk_errors]
            total_embedded += self._embedded
            reports.append(
                self._finish(
                    vault.slug, new, notes, [n.doc_id for n in notes], [], [], 0, errors, t0
                )
            )
        if self.settings.embeddings_enabled and total_with_chunks > 0 and total_embedded == 0:
            self.backend.delete_index(new)
            raise EmbeddingError(
                f"Full reindex produced 0 embedded notes across {total_with_chunks} chunked notes; "
                f"aborted alias swap and removed {new}."
            )
        self.backend.refresh(new)
        self.backend.swap_alias(old, new)
        return reports

    def _finish(
        self, slug, index, notes, new, changed, removed, unchanged, errors, t0
    ) -> SyncReport:
        report = SyncReport(
            vault=slug,
            scanned=len(notes),
            new=len(new),
            changed=len(changed),
            removed=len(removed),
            unchanged=unchanged,
            errors=errors[:20],
            warnings=sum(len(n.warnings) for n in notes),
            seconds=round(time.monotonic() - t0, 2),
            embedded=self._embedded,
            embed_seconds=round(self._embed_seconds, 2),
            index=index,
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.store.record_sync(slug, report.to_dict())
        return report
