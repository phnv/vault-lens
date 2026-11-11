from dataclasses import replace
from pathlib import Path

import pytest

import vaultlens.vault.parser as parser_mod
from vaultlens.config import Settings, VaultConfig, load_settings
from vaultlens.indexing.service import IndexCompatibilityError, IndexService, resolve_duplicate_ids
from vaultlens.search.mapping import check_meta_compatibility, index_name, version_of
from vaultlens.store import JsonStore
from vaultlens.vault.parser import parse_note

FIX = Path(__file__).parent / "fixtures" / "vault"


class _FakeBackend:
    def __init__(self, alias: str = "notes") -> None:
        self.alias = alias
        self.active_index: str | None = None
        self.meta_by_index: dict[str, dict] = {}
        self.docs_by_index: dict[str, dict[str, dict]] = {}

    def current_index(self) -> str | None:
        return self.active_index

    def check_index_compatibility(self, embedding_model: str = "off") -> tuple[bool, str]:
        if self.active_index is None:
            return True, "no index yet"
        return check_meta_compatibility(
            self.active_index,
            self.meta_by_index.get(self.active_index),
            embedding_model=embedding_model,
        )

    def ensure_index(self, embedding_model: str = "off") -> str:
        if self.active_index is None:
            self.active_index = index_name(self.alias, 1)
            self.meta_by_index[self.active_index] = {
                "mapping_version": 1,
                "embedding_model": embedding_model,
            }
            self.docs_by_index.setdefault(self.active_index, {})
        return self.active_index

    def create_next_index(self, embedding_model: str = "off") -> str:
        nxt = index_name(self.alias, version_of(self.active_index) + 1 if self.active_index else 1)
        self.meta_by_index[nxt] = {"mapping_version": 1, "embedding_model": embedding_model}
        self.docs_by_index[nxt] = {}
        return nxt

    def swap_alias(self, old: str | None, new: str) -> None:
        if old and old in self.docs_by_index and old != new:
            del self.docs_by_index[old]
        self.active_index = new

    def refresh(self, index: str | None = None) -> None:
        pass

    def indexed_hashes(self, vault: str) -> dict[str, str]:
        if self.active_index is None:
            return {}
        docs = self.docs_by_index.get(self.active_index, {})
        return {
            doc_id: src["content_hash"]
            for doc_id, src in docs.items()
            if src.get("vault") == vault
        }

    def bulk(self, actions):
        count = 0
        for act in actions:
            target = self.active_index if act["_index"] == self.alias else act["_index"]
            store = self.docs_by_index.setdefault(target, {})
            if act["_op_type"] == "index":
                store[act["_id"]] = act["_source"]
            elif act["_op_type"] == "delete":
                store.pop(act["_id"], None)
            count += 1
        return count, []


def test_duplicate_frontmatter_ids_fall_back_to_path_ids():
    a = parse_note("v", "a.md", "---\nid: 1\n---\nx")
    b = parse_note("v", "b.md", "---\nid: 1\n---\ny")
    resolve_duplicate_ids([b, a])
    assert a.doc_id == "v:1" and b.doc_id == "v:path:b.md" and b.warnings


def test_settings_from_env_and_yaml(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "c.yaml"
        cfg.write_text(
            "vaults:\n  - {slug: s, path: /v, ignore: [x]}\n"
            "index: {alias: notes, parser: default, chunker: heading, "
            "chunk_max_chars: 900, chunk_overlap: 50, embedding_model: off}\n"
        )
        s = load_settings({
            "CONFIG_PATH": str(cfg),
            "ES_URL": "http://h:9200",
            "CHUNK_OVERLAP": "",  # empty env var falls back to YAML
            "GEMINI_API_KEY": "SECRET-KEY-123",
            "GROQ_API_KEY": "GROQ-SECRET-456",
            "GROQ_MODEL": "qwen/qwen3.8-27b",
            "LLM_DEFAULT_PROVIDER": "ollama",
        })
    assert s.vault("s").ignore == ("x",) and s.chunk_max_chars == 900 and s.chunk_overlap == 50
    assert s.parser == "default" and s.chunker == "heading" and s.embedding_model == "off"
    assert s.es_url == "http://h:9200" and s.llm.default_provider == "ollama"
    assert s.llm.groq_model == "qwen/qwen3.8-27b"
    assert "SECRET-KEY-123" not in repr(s) and "GROQ-SECRET-456" not in repr(s)


def test_sync_reindexes_notes_on_chunker_or_parser_version_change_and_force(tmp_path, monkeypatch):
    vault = VaultConfig(slug="fixture", path=str(FIX))
    settings = Settings(data_dir=str(tmp_path), vaults=(vault,), chunk_max_chars=1500)
    backend = _FakeBackend()
    store = JsonStore(str(tmp_path))
    svc = IndexService(backend, settings, store)

    # 1. First sync indexes all 3 fixture notes as new
    r1 = svc.sync("fixture")
    assert (r1.new, r1.changed, r1.unchanged) == (3, 0, 0)

    # 2. Second sync with identical config reports all 3 unchanged
    r2 = svc.sync("fixture")
    assert (r2.new, r2.changed, r2.unchanged) == (0, 0, 3)

    # 3. Changing chunk_max_chars invalidates content_hash and re-indexes all 3 notes in-place
    svc.settings = replace(settings, chunk_max_chars=500)
    r3 = svc.sync("fixture")
    assert (r3.new, r3.changed, r3.unchanged) == (0, 3, 0)

    # 4. Bumping PARSER_VERSION invalidates content_hash on next sync
    monkeypatch.setattr(parser_mod, "PARSER_VERSION", 2)
    r4 = svc.sync("fixture")
    assert (r4.new, r4.changed, r4.unchanged) == (0, 3, 0)

    # 5. Force sync re-indexes all notes even when nothing changed
    r5_unchanged = svc.sync("fixture")
    assert r5_unchanged.unchanged == 3
    r5_forced = svc.sync("fixture", force=True)
    assert (r5_forced.new, r5_forced.changed, r5_forced.unchanged) == (0, 3, 0)


def test_sync_blocks_on_embedding_model_mismatch_until_full_reindex(tmp_path):
    vault = VaultConfig(slug="fixture", path=str(FIX))
    settings = Settings(data_dir=str(tmp_path), vaults=(vault,), embedding_model="off")
    backend = _FakeBackend()
    store = JsonStore(str(tmp_path))
    svc = IndexService(backend, settings, store)

    svc.sync("fixture")
    assert backend.current_index() == "notes_v1"

    # Switching embedding_model blocks incremental sync and instructs to run full_reindex
    svc.settings = replace(settings, embedding_model="bge-m3")
    with pytest.raises(IndexCompatibilityError, match="embedding_model"):
        svc.sync("fixture")

    # Full reindex builds notes_v2 with new _meta and swaps the alias, unblocking subsequent syncs
    reports = svc.full_reindex()
    assert len(reports) == 1 and reports[0].index == "notes_v2"
    assert backend.current_index() == "notes_v2"
    r_after = svc.sync("fixture")
    assert r_after.unchanged == 3
