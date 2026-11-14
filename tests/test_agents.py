from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from vaultlens.agents.agentic import run_agentic_search, validate_agent_filters
from vaultlens.agents.ai_search import dedupe_queries, run_ai_search
from vaultlens.agents.llm import get_chat_model, invoke_structured_with_repair
from vaultlens.agents.prompts import NO_RESULTS
from vaultlens.agents.retriever import HybridRetriever
from vaultlens.agents.schemas import AgentPlan, Grade, QueryPlan
from vaultlens.agents.trace import TraceSink
from vaultlens.config import AgentConfig, HybridConfig, LLMConfig, RerankConfig
from vaultlens.llm.base import LLMError
from vaultlens.models import SearchSpec
from vaultlens.search.hybrid import HybridSearcher
from vaultlens.search.tiers import evaluate_tier_status, get_tier


# --- Fakes --------------------------------------------------------------------------------
def _es_hit(doc_id: str, title: str, text: str, heading: str = "H0", pos: int = 0):
    return {
        "_id": doc_id,
        "_score": 1.0,
        "_source": {"title": title, "path": f"{doc_id}.md", "vault": "v"},
        "inner_hits": {
            "chunks": {
                "hits": {"hits": [{"_source": {"position": pos, "heading": heading, "text": text}}]}
            }
        },
    }


def _make_searcher(hits_by_query: dict[str, list[dict[str, Any]]]) -> HybridSearcher:
    def search_fn(body: dict[str, Any]) -> dict[str, Any]:
        if "knn" in body:
            kflt = body["knn"].get("filter", {})
            kclauses = kflt.get("bool", {}).get("filter", [kflt]) if kflt else []
            for clause in kclauses:
                if clause.get("terms", {}).get("status") == ["archived"]:
                    return {"took": 1, "hits": {"total": {"value": 0}, "hits": []}}
            all_hits = next(iter(hits_by_query.values()), [])
            return {"took": 1, "hits": {"total": {"value": len(all_hits)}, "hits": all_hits}}
        # lexical leg: inspect query string
        mm = body.get("query", {}).get("bool", {}).get("must", [{}])[0].get("multi_match", {})
        q = mm.get("query", "")
        flt = body.get("query", {}).get("bool", {}).get("filter", [])
        # If agent filter status=archived is present, simulate 0 hits to test auto-relaxation
        for clause in flt:
            terms = clause.get("terms", {})
            if terms.get("status") == ["archived"]:
                return {"took": 1, "hits": {"total": {"value": 0}, "hits": []}}
        hits = hits_by_query.get(q)
        if hits is None:
            hits = next(iter(hits_by_query.values()), [])
        return {"took": 1, "hits": {"total": {"value": len(hits)}, "hits": hits}}

    class _Emb:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    return HybridSearcher(
        search_fn,
        _Emb(),
        None,
        cfg=HybridConfig(candidates=5, rrf_k=60),
        rerank_cfg=RerankConfig(enabled=False),
    )


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _ScriptedChatModel:
    """Fake LangChain chat model supporting both `.invoke` and `.with_structured_output`."""

    def __init__(
        self,
        structured_outputs: list[Any] | None = None,
        text_outputs: list[Any] | None = None,
    ) -> None:
        self.structured_outputs = list(structured_outputs or [])
        self.text_outputs = list(text_outputs or [])
        self.structured_calls = 0
        self.text_calls = 0

    def with_structured_output(self, schema):
        parent = self

        class _Runner:
            def invoke(self, _messages):
                parent.structured_calls += 1
                if not parent.structured_outputs:
                    raise RuntimeError("no scripted structured output left")
                nxt = parent.structured_outputs.pop(0)
                if isinstance(nxt, Exception):
                    raise nxt
                if isinstance(nxt, dict):
                    return schema.model_validate(nxt)
                return nxt

        return _Runner()

    def invoke(self, _messages):
        self.text_calls += 1
        if not self.text_outputs:
            return _FakeMessage("Cited answer [1].")
        nxt = self.text_outputs.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _FakeMessage(str(nxt))


# --- llm.py & tier_status -----------------------------------------------------------------
def test_get_chat_model_enforces_groq_and_gemini_only():
    cfg = LLMConfig(groq_api_key="", gemini_api_key="")
    with pytest.raises(LLMError, match="off"):
        get_chat_model("off", cfg)
    with pytest.raises(LLMError, match="not supported"):
        get_chat_model("ollama", cfg)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        get_chat_model("groq", cfg)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        get_chat_model("gemini", cfg)


def test_invoke_structured_with_repair_retries_once_then_fails():
    # First fails, second succeeds -> 2 calls
    m_ok = _ScriptedChatModel(
        structured_outputs=[ValueError("bad json"), QueryPlan(rewrites=["r1"], sub_questions=[])]
    )
    out, calls = invoke_structured_with_repair(m_ok, QueryPlan, "sys", "usr")
    assert calls == 2 and out.rewrites == ["r1"]

    # Both fail -> raises LLMError
    m_bad = _ScriptedChatModel(structured_outputs=[ValueError("bad 1"), ValueError("bad 2")])
    with pytest.raises(LLMError, match="failed after repair retry"):
        invoke_structured_with_repair(m_bad, QueryPlan, "sys", "usr")


def test_evaluate_tier_status_reasons():
    ok_s, _ = evaluate_tier_status(get_tier("simple"), llm_provider="off")
    assert ok_s

    ok_h, reason_h = evaluate_tier_status(
        get_tier("hybrid"),
        llm_provider="off",
        config_embedding_model="off",
        live_embedding_model="off",
    )
    assert not ok_h and "index.embedding_model: off" in reason_h

    ok_h2, reason_h2 = evaluate_tier_status(
        get_tier("hybrid"),
        llm_provider="off",
        config_embedding_model="BAAI/bge-m3",
        live_embedding_model="off",
    )
    assert not ok_h2 and "no embeddings" in reason_h2
    ok_h3, reason_h3 = evaluate_tier_status(
        get_tier("hybrid"),
        llm_provider="off",
        config_embedding_model="BAAI/bge-m3",
        live_embedding_model="BAAI/bge-m3",
        total_chunks=753,
        embedded_chunks=0,
    )
    assert not ok_h3 and "0 embedded vectors" in reason_h3
    ok_h4, _ = evaluate_tier_status(
        get_tier("hybrid"),
        llm_provider="off",
        config_embedding_model="BAAI/bge-m3",
        live_embedding_model="BAAI/bge-m3",
        total_chunks=753,
        embedded_chunks=753,
    )
    assert ok_h4

    ok_ai, reason_ai = evaluate_tier_status(
        get_tier("ai_search"),
        llm_provider="ollama",
        config_embedding_model="BAAI/bge-m3",
        live_embedding_model="BAAI/bge-m3",
    )
    assert not ok_ai and "Groq or Gemini" in reason_ai

    ok_ai2, _ = evaluate_tier_status(
        get_tier("ai_search"),
        llm_provider="groq",
        config_embedding_model="BAAI/bge-m3",
        live_embedding_model="BAAI/bge-m3",
    )
    assert ok_ai2


# --- HybridRetriever (LCEL) ---------------------------------------------------------------
def test_hybrid_retriever_returns_documents_and_works_in_lcel():
    searcher = _make_searcher({"q": [_es_hit("v:1", "Doc One", "passage body", "Intro", 0)]})
    retriever = HybridRetriever(searcher=searcher, spec=SearchSpec(), top_n=5)
    chain = retriever | RunnableLambda(lambda docs: [d.metadata["doc_id"] for d in docs])
    res = chain.invoke("q")
    assert res == ["v:1"]
    docs = retriever.invoke("q")
    assert docs[0].page_content == "passage body"
    assert docs[0].metadata["title"] == "Doc One" and docs[0].metadata["position"] == 0


# --- Tier b (ai_search) -------------------------------------------------------------------
def test_dedupe_queries_preserves_original_first_and_caps_at_six():
    qs = dedupe_queries(
        "Orig Question",
        ["orig question", "Rewrite 1", "Rewrite 2", "Rewrite 3"],
        ["Sub 1", "Sub 2", "Sub 3", "Sub 4"],
    )
    assert qs[0] == "Orig Question"
    assert len(qs) == 6
    assert len({q.casefold() for q in qs}) == 6


def test_tier_b_graph_two_llm_calls_and_multi_query_merge():
    searcher = _make_searcher(
        {
            "main question": [_es_hit("v:1", "Note 1", "text from note 1")],
            "sub q1": [_es_hit("v:2", "Note 2", "text from note 2")],
        }
    )
    model = _ScriptedChatModel(
        structured_outputs=[
            QueryPlan(
                rewrites=["MAIN QUESTION", "pt translation"],
                sub_questions=["sub q1", "sub q2", "sub q3", "sub q4"],
            )
        ],
        text_outputs=["Combined answer citing [1] and [2]."],
    )
    sink = TraceSink()
    out = run_ai_search("main question", SearchSpec(), model, searcher, sink=sink)
    assert out.llm_calls == 2
    assert model.structured_calls == 1 and model.text_calls == 1
    assert out.queries[0] == "main question" and len(out.queries) == 6
    assert "Combined answer" in out.answer
    assert {s.doc_id for s in out.sources} == {"v:1", "v:2"}
    assert [e.step for e in sink.events if e.step in ("transform", "retrieve", "answer")] == [
        "transform",
        "retrieve",
        "answer",
    ]


def test_tier_b_no_passages_skips_answer_llm_and_llm_failure_degrades():
    empty_searcher = _make_searcher({"q": []})
    model = _ScriptedChatModel(
        structured_outputs=[QueryPlan(rewrites=[], sub_questions=[])],
        text_outputs=["should not be called"],
    )
    out_empty = run_ai_search("q", SearchSpec(), model, empty_searcher)
    assert out_empty.answer == NO_RESULTS
    assert model.text_calls == 0 and out_empty.llm_calls == 1

    # When both transform and answer LLM calls fail, tier degrades to hybrid results with warnings
    searcher = _make_searcher({"q": [_es_hit("v:1", "Note 1", "body 1")]})
    failing_model = _ScriptedChatModel(
        structured_outputs=[RuntimeError("down 1"), RuntimeError("down 2")],
        text_outputs=[RuntimeError("answer down")],
    )
    out_deg = run_ai_search("q", SearchSpec(), failing_model, searcher)
    assert len(out_deg.primary.hits) == 1
    assert out_deg.warnings and any("Answer generation failed" in w for w in out_deg.warnings)


# --- Tier c (agentic) ---------------------------------------------------------------------
def test_validate_agent_filters_drops_unknown_and_never_overrides_user_filters():
    allowed = {"type": ["doc", "meeting"], "status": ["active", "archived"], "tags": ["es"]}
    user = {"type": ["doc"]}
    proposed = {
        "type": ["meeting"],  # blocked: already set by user
        "status": ["active", "bogus"],  # 'bogus' dropped, 'active' kept
        "links_out": ["x"],  # disallowed field
        "unknown_field": ["y"],  # disallowed field
    }
    valid, dropped = validate_agent_filters(proposed, allowed, user)
    assert valid == {"status": ["active"]}
    assert len(dropped) == 4


def test_tier_c_filter_auto_relaxation_and_refine_loop():
    searcher = _make_searcher(
        {
            "q": [
                _es_hit("v:1", "N1", "passage 1"),
                _es_hit("v:2", "N2", "passage 2"),
                _es_hit("v:3", "N3", "passage 3"),
            ],
            "refined query": [_es_hit("v:4", "N4", "passage 4")],
        }
    )
    model = _ScriptedChatModel(
        structured_outputs=[
            # Plan proposes status=archived (which returns 0 hits in fake searcher -> auto-relaxed)
            AgentPlan(
                rewrites=[],
                sub_questions=["part B"],
                filters={"status": ["archived"]},
                rationale="check archived first",
            ),
            # Grade pass 1: insufficient -> triggers refine -> retrieve
            Grade(sufficient=False, missing=["part B"], next_queries=["refined query"]),
            # Grade pass 2: sufficient -> proceeds to synthesize
            Grade(sufficient=True, missing=["unanswered detail"], next_queries=[]),
        ],
        text_outputs=["Final synthesized answer [1] [4]; unanswered detail noted."],
    )
    sink = TraceSink()
    spec = SearchSpec(filters={"type": ["doc"]})
    out = run_agentic_search(
        "q",
        spec,
        model,
        searcher,
        cfg=AgentConfig(max_iterations=3, max_llm_calls=8, min_hits=3),
        allowed_facets={"status": ["archived", "active"], "type": ["doc"]},
        sink=sink,
    )
    steps = [e.step for e in sink.events]
    assert "relax_filters" in steps
    assert "refine" in steps
    assert out.iterations == 2
    assert out.llm_calls == 4  # plan(1) + grade1(1) + grade2(1) + synthesize(1)
    assert {p.doc_id for p in out.evidence} == {"v:1", "v:2", "v:3", "v:4"}
    # User filter survived the whole run
    retrieve_events = [e for e in sink.events if e.step == "retrieve"]
    for rev in retrieve_events:
        assert rev.data["filters"]["type"] == {"values": ["doc"], "source": "user"}


def test_tier_c_budget_termination_on_always_insufficient_model_and_repair_count():
    searcher = _make_searcher({"q": [_es_hit("v:1", "N1", "text 1")]})
    # Plan fails once then succeeds (2 calls); grade always says insufficient with new queries
    model = _ScriptedChatModel(
        structured_outputs=[
            ValueError("bad plan json"),
            AgentPlan(rewrites=[], sub_questions=[], filters={}, rationale="ok"),
            Grade(sufficient=False, missing=["x"], next_queries=["q2"]),
            Grade(sufficient=False, missing=["x"], next_queries=["q3"]),
            Grade(sufficient=False, missing=["x"], next_queries=["q4"]),
        ],
        text_outputs=["Bounded answer [1]."],
    )
    # Cap max_llm_calls=4: plan uses 2 (with repair), grade1 uses 1 (total 3),
    # then before/after grade2 budget triggers and routes straight to synthesize.
    out = run_agentic_search(
        "q",
        SearchSpec(),
        model,
        searcher,
        cfg=AgentConfig(max_iterations=5, max_llm_calls=4, timeout_s=60.0),
    )
    assert out.answer == "Bounded answer [1]."
    assert any("budget" in w.lower() for w in out.warnings)
    assert out.llm_calls <= 5
