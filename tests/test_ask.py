from vaultlens.ask import NO_RESULTS, SYSTEM_PROMPT, AskService, ask_spec, build_context, build_user_prompt
from vaultlens.llm.base import LLMResult
from vaultlens.models import SearchSpec
from vaultlens.search.query_builder import build_search_body

RAW = {
    "took": 1,
    "hits": {"total": {"value": 2}, "hits": [
        {"_id": "v:1", "_source": {"title": "A", "path": "a.md", "vault": "v"},
         "inner_hits": {"chunks": {"hits": {"hits": [{"_source": {"chunks": {"heading": "H", "position": 0, "text": "alpha text"}}}]}}}},
        {"_id": "v:2", "_source": {"title": "B", "path": "b.md", "vault": "v"},
         "highlight": {"body.pt": ["snippet <mark>beta</mark>"]}},
    ]},
}


class FakeLLM:
    provider, model = "fake", "m"

    def __init__(self):
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return LLMResult("answer [1]", "fake", "m", 1)

    def stream(self, system, user):
        yield "ans"

    def health(self):
        return True, "ok"


def test_ask_spec_is_keyword_retrieval_with_chunk_text():
    spec = ask_spec("what about X?", SearchSpec(vaults=["v"], filters={"tags": ["t"]}))
    assert spec.match_mode == "any" and spec.chunk_text and spec.with_chunks and not spec.facets
    assert spec.vaults == ["v"] and spec.filters == {"tags": ["t"]}  # user filters survive
    assert "chunks.text" in str(build_search_body(spec))


def test_context_numbers_sources_and_falls_back_to_snippet():
    from vaultlens.search.results import parse_response

    ctx, sources = build_context(parse_response(RAW).hits)
    assert [s.n for s in sources] == [1, 2]
    assert "[1] A (a.md) > H\nalpha text" in ctx and "snippet beta" in ctx and "<mark>" not in ctx


def test_answer_calls_llm_with_fenced_notes():
    llm = FakeLLM()
    result = AskService(lambda spec: RAW, llm).answer("q?")
    assert result.answer == "answer [1]" and len(result.sources) == 2
    system, user = llm.prompts[0]
    assert "untrusted" in system and user.startswith("<notes>") and user.endswith("Question: q?")
    assert build_user_prompt("q", "c") == "<notes>\nc\n</notes>\n\nQuestion: q" and SYSTEM_PROMPT


def test_no_hits_skips_llm():
    llm = FakeLLM()
    empty = {"hits": {"total": 0, "hits": []}}
    result = AskService(lambda spec: empty, llm).answer("q?")
    assert result.answer == NO_RESULTS and result.llm is None and llm.prompts == []
