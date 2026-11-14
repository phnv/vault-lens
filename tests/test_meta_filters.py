from datetime import date

import pytest

from vaultlens.models import SearchSpec
from vaultlens.search.query_builder import (
    build_search_body,
    date_range_spec,
    meta_filters,
    meta_rows,
    meta_values_body,
)
from vaultlens.search.results import parse_response


def test_meta_filters_builds_prefixed_terms_and_merges_keys():
    rows = [("course", ["ml"]), (" meta.course ", ["ml", "ai"]), ("prio", ["high"])]
    assert meta_filters(rows) == {"meta.course": ["ml", "ai"], "meta.prio": ["high"]}


def test_meta_filters_skips_blank_empty_and_invalid():
    rows = [("", ["x"]), ("course", []), ("bad key!", ["x"]), ("a.b", ["v"])]
    assert meta_filters(rows) == {"meta.a.b": ["v"]}


def test_meta_rows_round_trip_ignores_core_filters():
    filters = {"tags": ["t"], "meta.course": ["ml"], "meta.prio": []}
    assert meta_rows(filters) == [("course", ["ml"])]
    assert meta_filters(meta_rows(filters)) == {"meta.course": ["ml"]}


def test_meta_filter_reaches_search_body_as_terms():
    body = build_search_body(SearchSpec(filters=meta_filters([("course", ["ml", "ai"])])))
    assert {"terms": {"meta.course": ["ml", "ai"]}} in body["query"]["bool"]["filter"]


def test_meta_values_body_scopes_vault_and_validates_key():
    body = meta_values_body("course", "v1")
    assert body["size"] == 0
    assert body["aggs"]["values"]["terms"]["field"] == "meta.course"
    assert {"term": {"vault": "v1"}} in body["query"]["bool"]["filter"]
    assert meta_values_body("meta.course")["aggs"]["values"]["terms"]["field"] == "meta.course"
    with pytest.raises(ValueError):
        meta_values_body("bad key!")
    with pytest.raises(ValueError):
        meta_values_body("")


def test_values_agg_is_parsed_as_facet():
    raw = {
        "hits": {"total": 0, "hits": []},
        "aggregations": {"values": {"buckets": [{"key": "ml", "doc_count": 3}]}},
    }
    assert parse_response(raw).facets["values"] == [("ml", 3)]


def test_parse_hit_reads_created_and_indexed_at():
    raw = {
        "hits": {
            "total": 1,
            "hits": [
                {
                    "_id": "v:1",
                    "_score": 1.0,
                    "_source": {
                        "title": "T",
                        "path": "a.md",
                        "vault": "v",
                        "modified": "2026-01-02",
                        "created": "2025-12-31",
                        "indexed_at": "2026-01-03T00:00:00+00:00",
                    },
                }
            ],
        }
    }
    h = parse_response(raw).hits[0]
    assert (h.created, h.modified, h.indexed_at) == (
        "2025-12-31",
        "2026-01-02",
        "2026-01-03T00:00:00+00:00",
    )


def test_parse_hit_dates_optional():
    raw = {"hits": {"total": 1, "hits": [{"_id": "v:1", "_source": {"title": "T"}}]}}
    h = parse_response(raw).hits[0]
    assert h.created is None and h.indexed_at is None


def test_date_range_spec_variants_and_body():
    assert date_range_spec("created", []) is None
    assert date_range_spec("created", [date(2026, 1, 2)]) == {
        "field": "created",
        "gte": "2026-01-02",
    }
    dr = date_range_spec("modified", [date(2026, 1, 2), date(2026, 2, 3)])
    assert dr == {"field": "modified", "gte": "2026-01-02", "lte": "2026-02-03"}
    body = build_search_body(SearchSpec(date_range=dr))
    assert {"range": {"modified": {"gte": "2026-01-02", "lte": "2026-02-03"}}} in body["query"][
        "bool"
    ]["filter"]


def test_date_range_spec_rejects_unknown_field():
    with pytest.raises(ValueError):
        date_range_spec("indexed_at", [date(2026, 1, 2)])
