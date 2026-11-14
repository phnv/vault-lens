from dataclasses import replace

from vaultlens.models import SearchSpec
from vaultlens.search.query_builder import (
    ALLOWED_QUERY_FIELDS,
    STRATEGIES,
    build_search_body,
    inbound_links_body,
    strategy_spec,
)


def _mm(body):
    return body["query"]["bool"]["must"][0]["multi_match"]


def test_empty_query_is_match_all_with_filters_only():
    body = build_search_body(SearchSpec(vaults=["v"], filters={"status": ["active"]}))
    assert body["query"]["bool"]["must"] == [{"match_all": {}}]
    assert body["query"]["bool"]["filter"] == [
        {"terms": {"vault": ["v"]}},
        {"terms": {"status": ["active"]}},
    ]
    assert "should" not in body["query"]["bool"] and "highlight" not in body


def test_match_modes_map_to_operator_or_phrase():
    assert _mm(build_search_body(SearchSpec(query="a b", match_mode="all")))["operator"] == "and"
    any_mm = _mm(
        build_search_body(SearchSpec(query="a b", match_mode="any", minimum_should_match="75%"))
    )
    assert any_mm["operator"] == "or" and any_mm["minimum_should_match"] == "75%"
    ph = _mm(build_search_body(SearchSpec(query="a b", match_mode="phrase", fuzziness="AUTO")))
    assert ph["type"] == "phrase" and "operator" not in ph and "fuzziness" not in ph


def test_minimum_should_match_ignored_for_all_and_fuzziness_only_where_supported():
    mm = _mm(build_search_body(SearchSpec(query="a", match_mode="all", minimum_should_match="50%")))
    assert "minimum_should_match" not in mm
    cf = _mm(build_search_body(SearchSpec(query="a", match_type="cross_fields", fuzziness="AUTO")))
    assert "fuzziness" not in cf
    bf = _mm(build_search_body(SearchSpec(query="a", fuzziness="AUTO")))
    assert bf["fuzziness"] == "AUTO"


def test_field_boosts_rendered():
    mm = _mm(build_search_body(SearchSpec(query="a")))
    assert mm["fields"] == ["title^3", "aliases^2", "body.pt", "body.en"]


def test_chunk_clause_and_highlight_toggle():
    body = build_search_body(SearchSpec(query="a", chunk_text=True))
    nested = body["query"]["bool"]["should"][0]["nested"]
    assert nested["path"] == "chunks" and "chunks.text" in nested["inner_hits"]["_source"]
    assert "body.pt" in body["highlight"]["fields"]
    off = build_search_body(SearchSpec(query="a", with_chunks=False, highlight=False))
    assert "should" not in off["query"]["bool"] and "highlight" not in off


def test_sort_facets_explain_profile():
    body = build_search_body(
        SearchSpec(query="a", sort="modified_desc", explain=True, profile=True)
    )
    assert body["sort"][0] == {"modified": {"order": "desc", "missing": "_last"}}
    assert (
        body["explain"]
        and body["profile"]
        and set(body["aggs"]) == {"type", "status", "folder", "tags"}
    )
    assert "aggs" not in build_search_body(SearchSpec(facets=False))


def test_disallowed_fields_are_rejected():
    for bad in (
        SearchSpec(filters={"_id": ["x"]}),
        SearchSpec(query="a", fields={"password": 1.0}),
        SearchSpec(query="a", match_type="query_string"),
        SearchSpec(date_range={"field": "x", "gte": "now"}),
    ):
        try:
            build_search_body(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    ok = build_search_body(
        SearchSpec(filters={"meta.deadline": ["2026-12-01"], "folder.tree": ["10-Projects"]})
    )
    assert len(ok["query"]["bool"]["filter"]) == 2


def test_date_range_filter():
    body = build_search_body(SearchSpec(date_range={"field": "modified", "gte": "now-7d/d"}))
    assert body["query"]["bool"]["filter"] == [{"range": {"modified": {"gte": "now-7d/d"}}}]


def test_strategies_only_use_allowed_fields_and_are_note_level():
    for name in STRATEGIES:
        assert set(STRATEGIES[name]["fields"]) <= ALLOWED_QUERY_FIELDS
        body = build_search_body(strategy_spec(name, "q", ["v"]))
        assert (
            "should" not in body["query"]["bool"] and "aggs" not in body and "highlight" not in body
        )


def test_spec_roundtrip_ignores_unknown_keys():
    spec = replace(SearchSpec(), query="x", filters={"tags": ["a"]})
    data = spec.to_dict() | {"legacy_key": 1}
    assert SearchSpec.from_dict(data) == spec


def test_inbound_links_body_lowercases_and_excludes_self():
    body = inbound_links_body(
        "v", "30-Resources/Elasticsearch Analyzers.md", "ES Analyzers", ["ESA"]
    )
    terms = body["query"]["bool"]["filter"][1]["terms"]["links_out"]
    assert terms == [
        "30-resources/elasticsearch analyzers",
        "elasticsearch analyzers",
        "es analyzers",
        "esa",
    ]
    assert body["query"]["bool"]["must_not"] == [
        {"term": {"path": "30-Resources/Elasticsearch Analyzers.md"}}
    ]


def test_search_spec_defaults_match_mode_to_any():
    spec = SearchSpec(query="two words")
    assert spec.match_mode == "any"
    body = build_search_body(spec)
    assert body["query"]["bool"]["must"][0]["multi_match"]["operator"] == "or"
