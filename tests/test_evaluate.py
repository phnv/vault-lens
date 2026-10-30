from pathlib import Path

from vaultlens.evaluate import build_rank_eval_body, load_judgments, parse_rank_eval

FIX = Path(__file__).parent / "fixtures" / "judged_queries.yaml"


def test_load_and_build_rank_eval_body():
    judged = load_judgments(FIX)
    assert [q.id for q in judged] == ["q1"] and judged[0].ratings["fixture:202610031010"] == 3
    body = build_rank_eval_body(judged, "C", "notes_v2", k=10)
    req = body["requests"][0]
    assert body["metric"] == {"ndcg": {"k": 10, "normalize": True}}
    assert {"_index": "notes_v2", "_id": "fixture:202610031010", "rating": 3} in req["ratings"]
    assert "bool" in req["request"]["query"]


def test_strategies_produce_different_queries():
    judged = load_judgments(FIX)
    a, c = (build_rank_eval_body(judged, s, "i")["requests"][0]["request"] for s in "AC")
    assert a != c


def test_parse_rank_eval():
    r = parse_rank_eval("B", {"metric_score": 0.58, "details": {"q1": {"metric_score": 0.7}}})
    assert (r.strategy, r.label, r.mean, r.per_query) == ("B", "boosted fields", 0.58, {"q1": 0.7})


def test_invalid_rating_rejected(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "j.yaml"
        p.write_text("queries:\n  - id: q\n    query: x\n    relevant: [{doc_id: a, rating: 9}]\n")
        try:
            load_judgments(p)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
