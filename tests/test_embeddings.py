from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_indexing import _FakeBackend
from vaultlens.config import EmbedConfig, Settings, VaultConfig, load_settings
from vaultlens.indexing.service import IndexCompatibilityError, IndexService
from vaultlens.models import Chunk, Note
from vaultlens.search.mapping import MAPPING_VERSION, load_mapping, mapping_embedding_dims
from vaultlens.semantic.embedder import (
    EmbeddingError,
    OllamaEmbedder,
    SentenceTransformersEmbedder,
    chunk_embedding_text,
    l2_normalize,
    needs_query_prompt,
    validate_dims,
)
from vaultlens.store import JsonStore

FIX = Path(__file__).parent / "fixtures" / "vault"
DIMS = 1024


def _vec(x: float = 1.0) -> list[float]:
    return [x] + [0.0] * (DIMS - 1)


class FakeEmbedder:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.fail_on is not None and any(self.fail_on in t for t in texts):
            raise RuntimeError("boom")
        return [_vec() for _ in texts]

    def embed_query(self, text):
        return _vec()


def _svc(tmp_path, embedder, model="BAAI/bge-m3"):
    vault = VaultConfig(slug="fixture", path=str(FIX))
    settings = Settings(data_dir=str(tmp_path), vaults=(vault,), embedding_model=model)
    backend = _FakeBackend()
    return IndexService(backend, settings, JsonStore(str(tmp_path)), embedder), backend


# --- mapping v2 ---------------------------------------------------------------------------
def test_mapping_v2_has_vector_field_and_source_exclude():
    assert MAPPING_VERSION == 2
    m = load_mapping()
    emb = m["mappings"]["properties"]["chunks"]["properties"]["embedding"]
    assert emb == {"type": "dense_vector", "dims": DIMS, "index": True, "similarity": "cosine"}
    assert m["mappings"]["_source"] == {"excludes": ["chunks.embedding"]}
    assert m["mappings"]["dynamic"] == "strict"
    assert mapping_embedding_dims() == EmbedConfig().dims == DIMS


def test_validate_dims_rejects_mismatch():
    validate_dims(EmbedConfig())
    with pytest.raises(EmbeddingError):
        validate_dims(EmbedConfig(dims=768))


def test_to_es_doc_embeddings_optional_and_aligned():
    note = Note(
        vault="v",
        path="a.md",
        doc_id="v:1",
        title="t",
        body="b",
        content_hash="h",
        folder="",
        chunks=[Chunk(0, "H", 1, "x"), Chunk(1, "H2", 1, "y")],
    )
    assert all("embedding" not in c for c in note.to_es_doc()["chunks"])
    doc = note.to_es_doc([_vec(), _vec(0.5)])
    assert [c["embedding"][0] for c in doc["chunks"]] == [1.0, 0.5]
    with pytest.raises(ValueError):
        note.to_es_doc([_vec()])


# --- embedder -----------------------------------------------------------------------------
def test_chunk_embedding_text_header():
    assert chunk_embedding_text("Title", "Head", "body") == "Title > Head\nbody"
    assert chunk_embedding_text("", "Head", "body") == "Head\nbody"
    assert chunk_embedding_text("", "", "body") == "body"


def test_l2_normalize():
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class _FakeST:
    def __init__(self, name):
        self.name, self.kwargs = name, []
        self.max_seq_length = None

    def encode(self, texts, **kwargs):
        self.kwargs.append(kwargs)
        return [[0.0] * 3 for _ in texts]


def test_sentence_transformers_query_prompt_only_for_qwen():
    bge = SentenceTransformersEmbedder(EmbedConfig(), model_factory=_FakeST)
    bge.embed_query("q")
    assert "prompt_name" not in bge._model.kwargs[-1]
    assert bge._model.max_seq_length == 512

    qwen = SentenceTransformersEmbedder(
        EmbedConfig(model="Qwen/Qwen3-Embedding-0.6B"), model_factory=_FakeST
    )
    qwen.embed_query("q")
    assert qwen._model.kwargs[-1]["prompt_name"] == "query"
    qwen.embed_documents(["d"])
    assert "prompt_name" not in qwen._model.kwargs[-1]
    assert needs_query_prompt("qwen3-embedding:0.6b") and not needs_query_prompt("bge-m3")


def test_sentence_transformers_model_loads_lazily():
    loads = []
    emb = SentenceTransformersEmbedder(
        EmbedConfig(), model_factory=lambda n: loads.append(n) or _FakeST(n)
    )
    assert loads == []
    emb.embed_documents([])  # empty input never loads the model
    assert loads == []
    emb.embed_query("q")
    emb.embed_query("q")
    assert loads == ["BAAI/bge-m3"]


def test_ollama_embedder_failure_is_user_safe():
    emb = OllamaEmbedder(EmbedConfig(provider="ollama"), "http://127.0.0.1:9", timeout_s=1)
    with pytest.raises(EmbeddingError, match="Ollama embedding failed"):
        emb.embed_query("secret note text")


# --- indexing -----------------------------------------------------------------------------
def test_sync_embeds_only_new_and_changed(tmp_path):
    emb = FakeEmbedder()
    svc, backend = _svc(tmp_path, emb)
    r1 = svc.sync("fixture")
    assert r1.embedded == r1.new == 3 and len(emb.calls) == 3
    docs = backend.docs_by_index[backend.current_index()]
    assert all("embedding" in c for d in docs.values() for c in d["chunks"])

    emb.calls.clear()
    r2 = svc.sync("fixture")
    assert r2.unchanged == 3 and r2.embedded == 0 and emb.calls == []

    r3 = svc.sync("fixture", force=True)
    assert r3.embedded == 3


def test_embedding_texts_include_title_and_heading(tmp_path):
    emb = FakeEmbedder()
    svc, _ = _svc(tmp_path, emb)
    svc.sync("fixture")
    titles = {n.title for n in svc.load_notes(svc.settings.vault("fixture"))[0]}
    first_line = emb.calls[0][0].split("\n")[0]
    assert any(first_line.startswith(t) for t in titles)


def test_embedding_failure_does_not_abort_sync(tmp_path):
    svc, backend = _svc(tmp_path, FakeEmbedder(fail_on=""))  # every note fails
    r = svc.sync("fixture")
    assert r.new == 3 and r.embedded == 0
    assert len(r.errors) == 3 and all("embedding failed" in e for e in r.errors)
    docs = backend.docs_by_index[backend.current_index()]
    assert len(docs) == 3 and all("embedding" not in c for d in docs.values() for c in d["chunks"])


def test_wrong_vector_dims_recorded_as_failure(tmp_path):
    class Bad(FakeEmbedder):
        def embed_documents(self, texts):
            return [[1.0, 2.0] for _ in texts]

    svc, _ = _svc(tmp_path, Bad())
    r = svc.sync("fixture")
    assert r.embedded == 0 and r.errors


def test_embeddings_off_skips_embedder(tmp_path):
    emb = FakeEmbedder()
    svc, backend = _svc(tmp_path, emb, model="off")
    r = svc.sync("fixture")
    assert r.embedded == 0 and emb.calls == []


def test_switching_embed_model_blocks_sync_until_reindex(tmp_path):
    svc, backend = _svc(tmp_path, FakeEmbedder(), model="BAAI/bge-m3")
    svc.sync("fixture")
    svc.settings = replace(svc.settings, embedding_model="Qwen/Qwen3-Embedding-0.6B")
    with pytest.raises(IndexCompatibilityError, match="embedding_model"):
        svc.sync("fixture")
    reports = svc.full_reindex()
    assert reports[0].embedded == 3
    assert svc.sync("fixture").unchanged == 3


def test_sync_and_reindex_fail_fast_without_embedder(tmp_path):
    svc, backend = _svc(tmp_path, embedder=None, model="BAAI/bge-m3")
    with pytest.raises(EmbeddingError, match="no embedder"):
        svc.sync("fixture")
    assert backend.current_index() is None

    with pytest.raises(EmbeddingError, match="no embedder"):
        svc.full_reindex()
    assert backend.current_index() is None


def test_full_reindex_aborts_alias_swap_when_all_embeddings_fail(tmp_path):
    svc, backend = _svc(tmp_path, FakeEmbedder(), model="BAAI/bge-m3")
    svc.sync("fixture")
    live_before = backend.current_index()
    assert live_before == "notes_v1"

    svc.embedder = FakeEmbedder(fail_on="")  # every note fails embedding
    with pytest.raises(EmbeddingError, match="aborted alias swap"):
        svc.full_reindex()

    assert backend.current_index() == live_before
    assert "notes_v2" not in backend.docs_by_index


# --- config -------------------------------------------------------------------------------
def test_embedding_model_comes_only_from_yaml(tmp_path):
    off = load_settings({"CONFIG_PATH": str(tmp_path / "none.yaml")})
    assert off.embedding_model == "off" and not off.embeddings_enabled

    cfg = tmp_path / "c.yaml"
    cfg.write_text("index:\n  embedding_model: Qwen/Qwen3-Embedding-0.6B\n")
    on = load_settings(
        {
            "CONFIG_PATH": str(cfg),
            "EMBEDDING_MODEL": "off",
            "EMBED_MODEL": "other-model",
        }
    )
    assert on.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert on.embed.model == "Qwen/Qwen3-Embedding-0.6B"
    assert on.embeddings_enabled
