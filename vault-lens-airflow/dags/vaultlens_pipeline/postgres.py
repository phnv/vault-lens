"""PostgreSQL catalog operations for VaultLens notes."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from vaultlens.models import Note

logger = logging.getLogger(__name__)


def get_postgres_connection():
    """Connects to PostgreSQL inside docker network with localhost fallback."""
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is required for PostgreSQL catalog operations. Ensure it is installed."
        )

    host = os.getenv("POSTGRES_HOST", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "airflow")
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")

    try:
        return psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            connect_timeout=10,
        )
    except Exception as err:
        logger.warning("Could not connect to %s:%s (%s). Retrying localhost...", host, port, err)
        return psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host="localhost",
            port=port,
            connect_timeout=10,
        )


def check_or_create_table() -> str:
    """Checks and creates the vault_notes table and index in PostgreSQL if they do not exist."""
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vault_notes (
                    doc_id VARCHAR(255) PRIMARY KEY,
                    vault VARCHAR(64) NOT NULL,
                    path TEXT NOT NULL,
                    title TEXT,
                    content_hash VARCHAR(64) NOT NULL,
                    modified_at TIMESTAMPTZ,
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_vault_notes_vault ON vault_notes (vault);
            """)
        conn.commit()
        msg = "Table 'vault_notes' is ready in PostgreSQL."
        logger.info(msg)
        return msg
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to check/create vault_notes table: %s", exc)
        raise
    finally:
        conn.close()


def sync_note_catalog(vault: str, notes: list[Note]) -> dict[str, int]:
    """Upserts current notes into vault_notes and removes stale entries for the given vault."""
    conn = get_postgres_connection()
    upserted = 0
    deleted = 0

    try:
        with conn.cursor() as cur:
            if notes:
                records = [
                    (
                        n.doc_id,
                        n.vault,
                        n.path,
                        n.title,
                        n.content_hash,
                        n.modified,
                    )
                    for n in notes
                ]
                insert_sql = """
                    INSERT INTO vault_notes (doc_id, vault, path, title, content_hash, modified_at)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE SET
                        vault = EXCLUDED.vault,
                        path = EXCLUDED.path,
                        title = EXCLUDED.title,
                        content_hash = EXCLUDED.content_hash,
                        modified_at = EXCLUDED.modified_at,
                        indexed_at = CURRENT_TIMESTAMP;
                """
                if execute_values is not None:
                    execute_values(cur, insert_sql, records, page_size=200)
                else:
                    # Fallback for environments where execute_values is mocked
                    import vaultlens_pipeline.postgres as _self_mod
                    if _self_mod.execute_values is not None:
                        _self_mod.execute_values(cur, insert_sql, records, page_size=200)
                    else:
                        raise RuntimeError("execute_values unavailable")
                upserted = len(records)

                # Delete stale notes that exist in DB for this vault but no longer on disk
                active_ids = [n.doc_id for n in notes]
                cur.execute(
                    "DELETE FROM vault_notes WHERE vault = %s AND NOT (doc_id = ANY(%s));",
                    (vault, active_ids),
                )
                deleted = cur.rowcount if cur.rowcount != -1 else 0
            else:
                cur.execute("DELETE FROM vault_notes WHERE vault = %s;", (vault,))
                deleted = cur.rowcount if cur.rowcount != -1 else 0

        conn.commit()
        logger.info(
            "Catalog sync for vault '%s': %d upserted, %d pruned.", vault, upserted, deleted
        )
        return {"upserted": upserted, "deleted": deleted}
    except Exception as exc:
        conn.rollback()
        logger.error("Error synchronizing note catalog for vault '%s': %s", vault, exc)
        raise
    finally:
        conn.close()
