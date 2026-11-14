from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_indexing import _FakeBackend
from vaultlens.config import EmbedConfig, Settings, VaultConfig, load_settings
from vaultlens.indexing.service import IndexCompatibilityError, IndexService
from vaultlens.models import Chunk, Note
from vaultlens.search.mapping import MAPPING_VERSION, load_mapping, mapping_embedding_dims
from vaultlens.semantic.embedder import (
    QWEN3_QUERY_PREFIX,
    EmbeddingError,
    LangChainEmbedder,
    _build_fastembed,
    _build_hf_endpoint,
    _build_huggingface_local,
    _build_ollama,
    chunk_embedding_text,
    get_embedder,
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


class _FakeLCEmbeddings:
    def __init__(self, fail: bool = False) -> None:
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("boom secret_token_123")
        self.doc_calls.append(list(texts))
        return [[3.0, 4.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("boom secret_token_123")
        self.query_calls.append(text)
        return [3.0, 4.0]


def test_langchain_embedder_lazy_load_batching_and_normalization():
    loads = 0

    def _factory():
        nonlocal loads
        loads += 1
        return _FakeLCEmbeddings()

    cfg = EmbedConfig(provider="fastembed", batch=2)
    emb = LangChainEmbedder(cfg, _factory)
    assert loads == 0
    assert emb.embed_documents([]) == []  # empty input never loads the model
    assert loads == 0

    vecs = emb.embed_documents(["a", "b", "c"])
    assert loads == 1
    assert len(vecs) == 3 and vecs[0] == pytest.approx([0.6, 0.8])
    assert emb.embed_query("q") == pytest.approx([0.6, 0.8])
    assert loads == 1


def test_langchain_embedder_batches_documents_and_applies_query_prefix():
    fake = _FakeLCEmbeddings()
    cfg = EmbedConfig(provider="hf_inference", model="Qwen/Qwen3-Embedding-0.6B", batch=2)
    emb = LangChainEmbedder(
        cfg,
        lambda: fake,
        batch_documents=True,
        query_prefix=QWEN3_QUERY_PREFIX,
    )
    vecs = emb.embed_documents(["d1", "d2", "d3"])
    assert len(vecs) == 3
    assert fake.doc_calls == [["d1", "d2"], ["d3"]]

    emb.embed_query("search term")
    assert fake.query_calls == [f"{QWEN3_QUERY_PREFIX}search term"]
    assert needs_query_prompt("qwen3-embedding:0.6b") and not needs_query_prompt("bge-m3")


def test_langchain_embedder_failure_is_user_safe():
    cfg = EmbedConfig(provider="ollama")
    emb = LangChainEmbedder(cfg, lambda: _FakeLCEmbeddings(fail=True))
    with pytest.raises(EmbeddingError) as exc_info:
        emb.embed_query("secret note text")
    msg = str(exc_info.value)
    assert "ollama embedding failed" in msg
    assert "secret_token_123" not in msg and "secret note text" not in msg


def test_langchain_builders_wire_off_the_shelf_classes(monkeypatch):
    import langchain_community.embeddings.fastembed as fe_mod
    import langchain_huggingface as lchf_mod
    import langchain_ollama as lcol_mod

    fe_captured = {}

    class _StubFastEmbedEmbeddings:
        def __init__(self, **kw):
            fe_captured.update(kw)

    monkeypatch.setattr(fe_mod, "FastEmbedEmbeddings", _StubFastEmbedEmbeddings)
    _build_fastembed(
        EmbedConfig(provider="fastembed", model="BAAI/bge-m3", max_tokens=512, batch=8)
    )
    assert fe_captured == {"model_name": "BAAI/bge-m3", "max_length": 512, "batch_size": 8}

    # Missing HF_TOKEN raises EmbeddingError before constructing HuggingFaceEndpointEmbeddings
    with pytest.raises(EmbeddingError, match="HF_TOKEN is required"):
        _build_hf_endpoint(EmbedConfig(provider="hf_inference", hf_token=""))

    hf_ep_captured = {}

    class _StubHFEndpointEmbeddings:
        def __init__(self, **kw):
            hf_ep_captured.update(kw)

    monkeypatch.setattr(lchf_mod, "HuggingFaceEndpointEmbeddings", _StubHFEndpointEmbeddings)
    _build_hf_endpoint(
        EmbedConfig(provider="hf_inference", model="BAAI/bge-m3", hf_token="hf_tok_123")
    )
    assert hf_ep_captured == {
        "model": "BAAI/bge-m3",
        "provider": "hf-inference",
        "task": "feature-extraction",
        "huggingfacehub_api_token": "hf_tok_123",
    }

    hf_local_captured = {}

    class _StubHFEmbeddings:
        def __init__(self, **kw):
            hf_local_captured.update(kw)
            self._client = type("_Client", (), {"max_seq_length": None})()

    monkeypatch.setattr(lchf_mod, "HuggingFaceEmbeddings", _StubHFEmbeddings)
    st_qwen = _build_huggingface_local(
        EmbedConfig(
            provider="sentence_transformers",
            model="Qwen/Qwen3-Embedding-0.6B",
            max_tokens=512,
            batch=16,
        )
    )
    assert hf_local_captured["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert hf_local_captured["query_encode_kwargs"]["prompt_name"] == "query"
    assert st_qwen._client.max_seq_length == 512

    ol_captured = {}

    class _StubOllamaEmbeddings:
        def __init__(self, **kw):
            ol_captured.update(kw)

    monkeypatch.setattr(lcol_mod, "OllamaEmbeddings", _StubOllamaEmbeddings)
    _build_ollama(EmbedConfig(provider="ollama", model="bge-m3"), "http://localhost:11434/")
    assert ol_captured == {"model": "bge-m3", "base_url": "http://localhost:11434"}


def test_get_embedder_dispatches_all_providers():
    for prov in ("sentence_transformers", "fastembed", "hf_inference", "ollama"):
        s = Settings(embed=EmbedConfig(provider=prov, hf_token="hf_tok"))
        emb = get_embedder(s)
        assert isinstance(emb, LangChainEmbedder) and emb.cfg.provider == prov


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
