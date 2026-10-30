"""Vault -> index synchronisation. Incremental (hash diff) and full (build vN+1, swap alias)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import Settings, VaultConfig
from ..models import Note
from ..search.backend import SearchBackend
from ..store import JsonStore
from ..vault.parser import parse_note
from ..vault.scanner import diff_hashes, scan_vault


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
    index: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_duplicate_ids(notes: list[Note]) -> None:
    """Two notes sharing a frontmatter id would overwrite each other. Later path falls back to a path id."""
    seen: set[str] = set()
    for note in sorted(notes, key=lambda n: n.path):
        if note.doc_id in seen:
            note.warnings.append(f"duplicate frontmatter id {note.note_id!r}; using path-based doc id")
            note.doc_id = f"{note.vault}:path:{note.path}"
        seen.add(note.doc_id)


class IndexService:
    def __init__(self, backend: SearchBackend, settings: Settings, store: JsonStore) -> None:
        self.backend, self.settings, self.store = backend, settings, store

    def load_notes(self, vault: VaultConfig) -> tuple[list[Note], list[str]]:
        notes: list[Note] = []
        errors: list[str] = []
        for raw in scan_vault(vault):
            try:
                notes.append(parse_note(vault.slug, raw.path, raw.text, raw.mtime, self.settings.chunk_max_chars))
            except Exception as exc:  # one bad file must not abort a sync
                errors.append(f"{raw.path}: {type(exc).__name__}")
        resolve_duplicate_ids(notes)
        return notes, errors

    @staticmethod
    def _index_action(index: str, note: Note) -> dict[str, Any]:
        return {"_op_type": "index", "_index": index, "_id": note.doc_id, "_source": note.to_es_doc()}

    def sync(self, vault_slug: str) -> SyncReport:
        """Incremental: index new/changed notes, delete removed ones."""
        t0 = time.monotonic()
        vault = self.settings.vault(vault_slug)
        index = self.backend.ensure_index()
        notes, errors = self.load_notes(vault)
        by_id = {n.doc_id: n for n in notes}
        d = diff_hashes({i: n.content_hash for i, n in by_id.items()}, self.backend.indexed_hashes(vault.slug))
        actions = [self._index_action(self.backend.alias, by_id[i]) for i in d.new + d.changed]
        actions += [{"_op_type": "delete", "_index": self.backend.alias, "_id": i} for i in d.removed]
        if actions:
            _, bulk_errors = self.backend.bulk(actions)
            errors += [str(e)[:200] for e in bulk_errors]
            self.backend.refresh()
        return self._finish(vault.slug, index, notes, d.new, d.changed, d.removed, d.unchanged, errors, t0)

    def full_reindex(self) -> list[SyncReport]:
        """Build notes_vN+1 from every registered vault, then swap the alias atomically."""
        old = self.backend.current_index()
        new = self.backend.create_next_index()
        reports: list[SyncReport] = []
        for vault in self.settings.vaults:
            t0 = time.monotonic()
            notes, errors = self.load_notes(vault)
            _, bulk_errors = self.backend.bulk(self._index_action(new, n) for n in notes)
            errors += [str(e)[:200] for e in bulk_errors]
            reports.append(self._finish(vault.slug, new, notes, [n.doc_id for n in notes], [], [], 0, errors, t0))
        self.backend.refresh(new)
        self.backend.swap_alias(old, new)
        return reports

    def _finish(self, slug, index, notes, new, changed, removed, unchanged, errors, t0) -> SyncReport:
        report = SyncReport(
            vault=slug, scanned=len(notes), new=len(new), changed=len(changed), removed=len(removed),
            unchanged=unchanged, errors=errors[:20], warnings=sum(len(n.warnings) for n in notes),
            seconds=round(time.monotonic() - t0, 2), index=index,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.store.record_sync(slug, report.to_dict())
        return report
