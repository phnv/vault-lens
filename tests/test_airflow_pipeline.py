from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add airflow dags and src to path for testing vaultlens_pipeline
ROOT_DIR = Path(__file__).parent.parent
DAGS_PATH = ROOT_DIR / "vault-lens-airflow" / "dags"
SRC_PATH = ROOT_DIR / "src"

for p in (str(DAGS_PATH), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vaultlens.config import Settings, VaultConfig
from vaultlens.indexing.service import SyncReport
from vaultlens.models import Note

try:
    from vaultlens_pipeline.postgres import check_or_create_table, sync_note_catalog
    from vaultlens_pipeline.sync import sync_vault_notes
    HAVE_PIPELINE = True
except ImportError:
    HAVE_PIPELINE = False


class TestAirflowPipeline(unittest.TestCase):
    def test_dag_schedule_configuration(self):
        dag_file = DAGS_PATH / "vaultlens_sync_dag.py"
        if not dag_file.exists():
            self.skipTest(f"DAG file not found at {dag_file}")
        content = dag_file.read_text(encoding="utf-8")
        self.assertIn('schedule_interval="*/15 * * * *"', content)
        self.assertIn('catchup=False', content)
        self.assertIn('max_active_runs=1', content)
        self.assertIn('dag_id="vaultlens_sync_notes"', content)

    @unittest.skipUnless(HAVE_PIPELINE, "vaultlens_pipeline not found in path")
    def test_check_or_create_table(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch("vaultlens_pipeline.postgres.get_postgres_connection", return_value=mock_conn):
            result = check_or_create_table()

        self.assertIn("ready", result)
        self.assertTrue(mock_cur.execute.called)
        create_stmt = mock_cur.execute.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS vault_notes", create_stmt)
        self.assertIn("doc_id VARCHAR(255) PRIMARY KEY", create_stmt)
        self.assertIn("content_hash VARCHAR(64) NOT NULL", create_stmt)
        self.assertTrue(mock_conn.commit.called)
        self.assertTrue(mock_conn.close.called)

    @unittest.skipUnless(HAVE_PIPELINE, "vaultlens_pipeline not found in path")
    def test_sync_note_catalog_upsert_and_prune(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        notes = [
            Note(
                vault="personal-vault",
                path="note1.md",
                doc_id="personal-vault:path:note1.md",
                title="Note 1",
                body="Hello world",
                content_hash="hash123",
                folder="",
                modified="2026-10-27T12:00:00Z",
            )
        ]

        mock_exec_val = MagicMock()
        with patch("vaultlens_pipeline.postgres.get_postgres_connection", return_value=mock_conn), \
             patch("vaultlens_pipeline.postgres.execute_values", mock_exec_val):
            stats = sync_note_catalog("personal-vault", notes)

        self.assertEqual(stats, {"upserted": 1, "deleted": 1})
        self.assertTrue(mock_exec_val.called)
        args = mock_exec_val.call_args[0]
        records = args[2]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], "personal-vault:path:note1.md")
        self.assertEqual(records[0][1], "personal-vault")
        self.assertEqual(records[0][4], "hash123")
        self.assertTrue(mock_conn.commit.called)

    @unittest.skipUnless(HAVE_PIPELINE, "vaultlens_pipeline not found in path")
    def test_sync_vault_notes_orchestration(self):
        dummy_vault = VaultConfig(slug="test-vault", path="/test/vault")
        dummy_settings = Settings(
            es_url="http://mock-es:9200",
            es_alias="notes",
            data_dir="/tmp/test_data",
            vaults=(dummy_vault,),
        )
        dummy_report = SyncReport(
            vault="test-vault",
            scanned=5,
            new=2,
            changed=1,
            removed=0,
            index="notes_v1",
            seconds=0.15,
        )

        mock_svc = MagicMock()
        mock_svc.load_notes.return_value = ([], [])
        mock_svc.sync.return_value = dummy_report

        with patch("vaultlens_pipeline.sync.load_settings", return_value=dummy_settings), \
             patch("vaultlens_pipeline.sync.SearchBackend"), \
             patch("vaultlens_pipeline.sync.JsonStore"), \
             patch("vaultlens_pipeline.sync.IndexService", return_value=mock_svc), \
             patch("vaultlens_pipeline.sync.sync_note_catalog", return_value={"upserted": 5, "deleted": 0}):
            out = sync_vault_notes("test-vault")

        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["vaults"]), 1)
        res = out["vaults"][0]
        self.assertEqual(res["vault"], "test-vault")
        self.assertEqual(res["scanned"], 5)
        self.assertEqual(res["new"], 2)
        self.assertEqual(res["es_index"], "notes_v1")
        self.assertEqual(res["postgres_upserted"], 5)


if __name__ == "__main__":
    unittest.main()
