import pytest

from vaultlens.config import HybridConfig, RerankConfig
from vaultlens.models import SearchSpec, TraceEvent
from vaultlens.search.hybrid import HybridSearcher, knn_body, rrf_fuse
from vaultlens.semantic.reranker import CrossEncoderReranker, prepare_rerank_text


# --- rrf_fuse -----------------------------------------------------------------------------
def test_rrf_fuse_ordering_weights_ties_and_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []

    # Single list preserves order
    single = rrf_fuse([["b", "a"]], k=60)
    assert [doc_id for doc_id, _ in single] == ["b", "a"]

    # Consensus beats single leg; ties break deterministically by doc_id
    fused = rrf_fuse([["b", "a"], ["c", "a"]], k=60)
    assert fused[0][0] == "a"
    assert [doc_id for doc_id, _ in fused[1:]] == ["b", "c"]

    # Weights shift priority
    weighted = rrf_fuse([["b"], ["c"]], k=60, weights=[1.0, 2.0])
    assert [doc_id for doc_id, _ in weighted] == ["c", "b"]

    with pytest.raises(ValueError):
        rrf_fuse([["a"]], weights=[1.0, 2.0])


# --- knn_body -----------------------------------------------------------------------------
def test_knn_body_reuses_filter_clauses_and_rejects_disallowed_fields():
    spec = SearchSpec(
        query="q",
        vaults=["v1"],
        filters={"type": ["doc"], "tags": ["t"]},
    )
    body = knn_body([0.5, 0.5], spec, k=10, num_candidates=30, inner_size=2)
    assert body["knn"]["field"] == "chunks.embedding"
    assert body["knn"]["k"] == 10 and body["knn"]["num_candidates"] == 30
    assert body["knn"]["inner_hits"]["_source"] == [
        "chunks.heading",
        "chunks.position",
        "chunks.text",
    ]
    assert "bool" in body["knn"]["filter"]

    bad_spec = SearchSpec(filters={"not_allowed": ["x"]})
    with pytest.raises(ValueError, match="filter field not allowed"):
        knn_body([0.5], bad_spec, k=5, num_candidates=15)


# --- reranker input text (R1) -------------------------------------------------------------
def test_prepare_rerank_text_full_chunk_and_oversized_fallback():
    # Normal chunk (<= 1600 chars) keeps full heading + text
    normal = prepare_rerank_text(
        "query", "Section", "Short chunk text.", snippet="<mark>Short</mark>"
    )
    assert normal == "Section\nShort chunk text."

    # Oversized chunk with lexical highlight falls back to stripped highlight snippet
    long_text = "x" * 2000
    with_hl = prepare_rerank_text(
        "query", "Head", long_text, snippet="found <mark>answer</mark> here"
    )
    assert with_hl == "Head\nfound answer here"

    # Oversized kNN-only chunk (no highlight) uses head truncation
    knn_only = prepare_rerank_text("query", "Head", long_text, snippet="")
    assert knn_only.startswith("Head\n" + "x" * 100)
    assert len(knn_only) <= 1600 + len("Head\n")


def test_cross_encoder_reranker_loads_lazily():
    loads = []

    class _FakeCE:
        def __init__(self, name):
            loads.append(name)

        def predict(self, pairs):
            return [float(len(p[1])) for p in pairs]

    rr = CrossEncoderReranker(RerankConfig(), model_factory=_FakeCE)
    assert loads == []
    assert rr.score_pairs("q", []) == []
    assert loads == []
    scores = rr.score_pairs("q", ["short", "longer text"])
    assert loads == [RerankConfig().model]
    assert scores == [5.0, 11.0]


# --- HybridSearcher -----------------------------------------------------------------------
def _es_hit(doc_id: str, title: str, chunks: list[tuple[int, str, str, str]]):
    inner = [
        {
            "_source": {"position": pos, "heading": head, "text": txt},
            **({"highlight": {"chunks.text.pt": [snip]}} if snip else {}),
        }
        for pos, head, txt, snip in chunks
    ]
    return {
        "_id": doc_id,
        "_score": 1.0,
        "_source": {"title": title, "path": f"{doc_id}.md", "vault": "v"},
        "inner_hits": {"chunks": {"hits": {"hits": inner}}},
    }


def _es_resp(hits, facets=None):
    return {
        "took": 4,
        "hits": {"total": {"value": len(hits)}, "hits": hits},
        "aggregations": facets or {},
    }


class _FakeEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        if self.fail:
            raise RuntimeError("embed down")
        return [1.0, 0.0]


class _FakeReranker:
    def __init__(self, scores_by_sub: dict[str, float] | None = None, fail: bool = False):
        self.scores_by_sub = scores_by_sub or {}
        self.fail = fail
        self.seen: list[str] = []

    def score_pairs(self, query, texts):
        if self.fail:
            raise RuntimeError("rerank down")
        self.seen = list(texts)
        out = []
        for t in texts:
            sc = 0.0
            for sub, val in self.scores_by_sub.items():
                if sub in t:
                    sc = val
            out.append(sc)
        return out


def test_hybrid_searcher_fusion_dedupe_and_blend_vs_replace():
    lex_hits = [
        _es_hit("v:1", "One", [(0, "H0", "alpha text", "<mark>alpha</mark>")]),
        _es_hit("v:2", "Two", [(0, "H0", "beta text", "")]),
    ]
    sem_hits = [
        _es_hit("v:2", "Two", [(0, "H0", "beta text", ""), (1, "H1", "beta second", "")]),
        _es_hit("v:3", "Three", [(0, "H0", "gamma winning text", "")]),
    ]

    def search_fn(body):
        if "knn" in body:
            return _es_resp(sem_hits)
        return _es_resp(lex_hits, {"type": {"buckets": [{"key": "doc", "doc_count": 2}]}})

    events: list[TraceEvent] = []
    reranker = _FakeReranker({"gamma": 10.0, "beta": 2.0, "alpha": 1.0})

    # Replace mode puts gamma first
    hs_replace = HybridSearcher(
        search_fn,
        _FakeEmbedder(),
        reranker,
        HybridConfig(candidates=10, rrf_k=60),
        RerankConfig(enabled=True, mode="replace", pool=10),
        trace=events.append,
    )
    resp_rep = hs_replace.search("test query", SearchSpec(size=5), top_n=4)
    assert resp_rep.passages[0].doc_id == "v:3"
    assert resp_rep.hits[0].doc_id == "v:3"
    assert resp_rep.facets["type"] == [("doc", 2)]
    assert resp_rep.legs == {"lexical": 2, "semantic": 2}
    # Duplicate chunk (v:2 pos 0 in both legs) appears only once
    v2_positions = [p.position for p in resp_rep.passages if p.doc_id == "v:2"]
    assert v2_positions == sorted(set(v2_positions))
    assert [e.step for e in events] == ["lexical", "semantic", "rerank"]

    # Blend mode combines initial RRF rank (where v:2 is #1) with reranker rank
    hs_blend = HybridSearcher(
        search_fn,
        _FakeEmbedder(),
        reranker,
        HybridConfig(candidates=10, rrf_k=60),
        RerankConfig(enabled=True, mode="blend", pool=10),
    )
    resp_blend = hs_blend.search("test query", SearchSpec(size=5), top_n=4)
    assert {p.doc_id for p in resp_blend.passages} == {"v:1", "v:2", "v:3"}


def test_hybrid_searcher_empty_query_and_degradation_paths():
    lex_hits = [_es_hit("v:1", "One", [(0, "H0", "alpha", "")])]

    def search_fn(body):
        if "knn" in body:
            return _es_resp([])
        return _es_resp(lex_hits)

    # Empty query skips kNN and rerank
    hs = HybridSearcher(search_fn, _FakeEmbedder(fail=True), _FakeReranker(fail=True))
    empty = hs.search("   ", SearchSpec())
    assert empty.legs == {"lexical": 1, "semantic": 0}
    assert len(empty.hits) == 1 and empty.passages == []

    # Embedder failure degrades to lexical leg with warning
    deg_sem = hs.search("hello", SearchSpec())
    assert len(deg_sem.hits) == 1
    assert any("Semantic leg failed" in w for w in deg_sem.warnings)
    assert any("Reranker failed" in w for w in deg_sem.warnings)

    # Both legs failing raises RuntimeError
    def fail_search(_body):
        raise RuntimeError("es down")

    hs_both_bad = HybridSearcher(fail_search, _FakeEmbedder(fail=True), None)
    with pytest.raises(RuntimeError, match="Both hybrid legs failed"):
        hs_both_bad.search("hello", SearchSpec())


def test_hybrid_searcher_synthesizes_fallback_passages_without_chunks():
    no_chunk_hits = [
        _es_hit("v:1", "Title Hit", [])
        | {"highlight": {"body.pt": ["<mark>Title</mark> match body"]}},
        _es_hit("v:2", "Only Title Hit", []),
    ]

    def search_fn(body):
        if "knn" in body:
            return _es_resp([])
        return _es_resp(no_chunk_hits)

    hs = HybridSearcher(
        search_fn,
        embedder=None,
        reranker=None,
        rerank_cfg=RerankConfig(enabled=False, mode="blend", pool=10),
    )
    resp = hs.search("title", SearchSpec(size=5), top_n=5)
    assert [p.doc_id for p in resp.passages] == ["v:1", "v:2"]
    assert resp.passages[0].text == "Title match body"
    assert resp.passages[0].position == -1
    assert resp.passages[1].text == "Only Title Hit"
    assert len(resp.hits) == 2
