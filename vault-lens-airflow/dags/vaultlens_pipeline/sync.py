"""Orchestrates VaultLens note loading, Elasticsearch indexing, and PostgreSQL catalog sync."""
from __future__ import annotations

import logging
from typing import Any

from vaultlens.config import load_settings
from vaultlens.indexing.service import IndexService
from vaultlens.search.backend import SearchBackend
from vaultlens.store import JsonStore

from .postgres import sync_note_catalog

logger = logging.getLogger(__name__)


def sync_vault_notes(vault_slug: str | None = None) -> dict[str, Any]:
    """Synchronizes markdown vaults into Elasticsearch and PostgreSQL.

    1. Scans notes and parses frontmatter, headings, links, and content hashes.
    2. Synchronizes incrementally into Elasticsearch (notes alias -> notes_v1).
    3. Upserts metadata catalog into PostgreSQL vault_notes table.
    4. Updates sync history in JsonStore for Streamlit UI visibility.
    """
    settings = load_settings()
    backend = SearchBackend(settings.es_url, settings.es_alias)
    store = JsonStore(settings.data_dir)
    svc = IndexService(backend, settings, store)

    vaults = [settings.vault(vault_slug)] if vault_slug else list(settings.vaults)
    if not vaults:
        logger.warning("No vaults registered in settings.")
        return {"status": "no_vaults_found", "vaults": []}

    summaries = []
    for vault in vaults:
        logger.info("Starting synchronization for vault '%s' at path '%s'", vault.slug, vault.path)
        notes, load_errors = svc.load_notes(vault)
        if load_errors:
            logger.warning("Encountered %d note parsing errors. First error: %s", len(load_errors), load_errors[0])

        report = svc.sync(vault.slug)
        pg_stats = sync_note_catalog(vault.slug, notes)

        summary = {
            "vault": vault.slug,
            "path": vault.path,
            "es_index": report.index,
            "scanned": report.scanned,
            "new": report.new,
            "changed": report.changed,
            "removed": report.removed,
            "seconds": report.seconds,
            "postgres_upserted": pg_stats.get("upserted", 0),
            "postgres_deleted": pg_stats.get("deleted", 0),
            "load_errors_count": len(load_errors),
        }
        summaries.append(summary)
        logger.info(
            "Sync complete for '%s': ES +%d new, %d changed, %d removed in %.2fs. Postgres: %d upserted.",
            vault.slug,
            report.new,
            report.changed,
            report.removed,
            report.seconds,
            pg_stats.get("upserted", 0),
        )

    return {"status": "success", "vaults": summaries}
