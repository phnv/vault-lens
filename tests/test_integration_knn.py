"""Nested-kNN check for ADR-0009 section 4 item 3. Needs a live Elasticsearch (skipped otherwise).

UNVERIFIED until run: proves nested `chunks.embedding` kNN with parent-level filters, `inner_hits`
chunk text, and the `_meta` round-trip, using fake vectors (no model needed).
"""

from uuid import uuid4

import pytest

from vaultlens.models import Chunk, Note
from vaultlens.search.mapping import load_mapping

from .conftest import es_url

pytestmark = pytest.mark.integration
DIMS = 1024


def _vec(i: int) -> list[float]:
    v = [0.0] * DIMS
    v[i] = 1.0
    return v


@pytest.fixture(scope="module")
def backend_with_vectors():
    from vaultlens.search.backend import SearchBackend

    alias = f"test_{uuid4().hex[:8]}"
    be = SearchBackend(es_url(), alias)
    name = f"{alias}_v2"
    be.es.indices.create(index=name, body=load_mapping(embedding_model="fake-model"))
    be.es.indices.put_alias(index=name, name=alias)
    notes = [
        Note(
            vault="v1",
            path="a.md",
            doc_id="v1:a",
            title="A",
            body="alpha",
            content_hash="1",
            folder="",
            type="doc",
            chunks=[Chunk(0, "H0", 1, "chunk zero"), Chunk(1, "H1", 1, "chunk one")],
        ),
        Note(
            vault="v2",
            path="b.md",
            doc_id="v2:b",
            title="B",
            body="beta",
            content_hash="2",
            folder="",
            type="doc",
            chunks=[Chunk(0, "H0", 1, "other zero")],
        ),
    ]
    vecs = {"v1:a": [_vec(0), _vec(1)], "v2:b": [_vec(0)]}
    actions = [
        {
            "_op_type": "index",
            "_index": name,
            "_id": n.doc_id,
            "_source": n.to_es_doc(vecs[n.doc_id]),
        }
        for n in notes
    ]
    be.bulk(actions)
    be.refresh()
    yield be
    be.es.indices.delete(index=name)


def _knn(vec, filt=None):
    knn = {
        "field": "chunks.embedding",
        "query_vector": vec,
        "k": 5,
        "num_candidates": 15,
        "inner_hits": {"size": 2, "_source": ["chunks.heading", "chunks.position", "chunks.text"]},
    }
    if filt:
        knn["filter"] = filt
    return {"knn": knn, "_source": ["title", "path"], "size": 5}


def test_nested_knn_returns_inner_hit_chunk_text(backend_with_vectors):
    r = backend_with_vectors.raw_search(_knn(_vec(1)))
    top = r["hits"]["hits"][0]
    assert top["_id"] == "v1:a"
    # verified against ES 8.17: nested inner_hits _source is the chunk object itself (unwrapped)
    inner = top["inner_hits"]["chunks"]["hits"]["hits"][0]["_source"]
    assert inner["text"] == "chunk one"


def test_nested_knn_parent_filter(backend_with_vectors):
    r = backend_with_vectors.raw_search(_knn(_vec(0), {"term": {"vault": "v2"}}))
    assert [h["_id"] for h in r["hits"]["hits"]] == ["v2:b"]


def test_meta_roundtrip_and_vector_not_in_source(backend_with_vectors):
    assert backend_with_vectors.embedding_meta() == {"embedding_model": "fake-model"}
    st = backend_with_vectors.stats()
    assert st["chunks"] > 0 and st["embedded_chunks"] == st["chunks"]
    src = backend_with_vectors.es.get(index=backend_with_vectors.alias, id="v1:a")["_source"]
    assert all("embedding" not in c for c in src["chunks"])
