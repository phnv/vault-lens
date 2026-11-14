import pytest

from vaultlens.config import load_settings
from vaultlens.models import HybridResponse, Passage, TraceEvent
from vaultlens.search.tiers import TIERS, get_tier


def _env(tmp_path, **kw):
    return {"CONFIG_PATH": str(tmp_path / "none.yaml"), **kw}


def test_tiers_order_and_flags():
    assert [t.id for t in TIERS] == ["simple", "ask", "hybrid", "ai_search", "agentic"]
    assert not get_tier("simple").needs_llm and not get_tier("simple").needs_embeddings
    assert get_tier("ask").needs_llm and not get_tier("ask").needs_embeddings
    assert get_tier("hybrid").needs_embeddings and not get_tier("hybrid").needs_llm
    with pytest.raises(KeyError):
        get_tier("nope")


def test_semantic_defaults(tmp_path):
    s = load_settings(_env(tmp_path))
    assert s.embed.model == "BAAI/bge-m3" and s.embed.dims == 1024
    assert s.rerank.enabled and s.rerank.mode == "blend" and s.rerank.pool == 20
    assert s.hybrid.candidates == 30 and s.hybrid.rrf_k == 60
    assert s.agent.max_iterations == 3 and s.agent.max_llm_calls == 8


def test_env_beats_yaml_beats_default(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("semantic:\n  rerank: {mode: replace, pool: 10}\n  hybrid: {rrf_k: 40}\n")
    s = load_settings({"CONFIG_PATH": str(cfg), "RERANK_POOL": "5", "RERANK_ENABLED": "false"})
    assert s.rerank.mode == "replace" and s.rerank.pool == 5 and not s.rerank.enabled
    assert s.hybrid.rrf_k == 40


def test_invalid_values_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_settings(_env(tmp_path, RERANK_MODE="bogus"))
    with pytest.raises(ValueError):
        load_settings(_env(tmp_path, EMBED_PROVIDER="bogus"))


def test_new_models_are_plain_and_serialisable():
    ev = TraceEvent(ts="t", step="x", kind="search", summary="s", ms=3, data={"a": 1})
    assert ev.to_dict()["data"] == {"a": 1}
    p = Passage("d", "a.md", "T", "H", "v", 0, "text")
    assert HybridResponse(passages=[p], hits=[]).passages[0].rerank_score is None
