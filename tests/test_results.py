from vaultlens.search.results import parse_response, strip_marks

RAW = {
    "took": 12,
    "hits": {
        "total": {"value": 37, "relation": "eq"},
        "hits": [
            {
                "_id": "v:1",
                "_score": 2.5,
                "_source": {
                    "title": "T",
                    "path": "a/b.md",
                    "vault": "v",
                    "tags": ["x"],
                    "links_out": ["n"],
                    "aliases": ["al"],
                },
                "highlight": {"body.pt": ["combine a <mark>analyzer</mark> &amp; more"]},
                "inner_hits": {
                    "chunks": {
                        "hits": {
                            "hits": [
                                {
                                    "_source": {
                                        "chunks": {
                                            "heading": "Analyzer choice",
                                            "position": 2,
                                            "text": "body",
                                        }
                                    },
                                    "highlight": {"chunks.text.pt": ["<mark>analyzer</mark>"]},
                                }
                            ]
                        }
                    }
                },
            }
        ],
    },
    "aggregations": {"type": {"buckets": [{"key": "atomic", "doc_count": 24}]}},
}


def test_parse_response():
    r = parse_response(RAW)
    assert (r.total, r.took_ms) == (37, 12)
    h = r.hits[0]
    assert (h.doc_id, h.title, h.links_out, h.aliases) == ("v:1", "T", ["n"], ["al"])
    assert h.chunks[0].heading == "Analyzer choice" and h.chunks[0].position == 2
    assert r.facets == {"type": [("atomic", 24)]}


def test_strip_marks_unescapes():
    assert strip_marks("a <mark>b</mark> &amp; c") == "a b & c"


def test_empty_response():
    assert parse_response({"hits": {"total": 0, "hits": []}}).total == 0
