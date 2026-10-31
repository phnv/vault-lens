"""Saved-search helpers: matching the loaded saved spec and deleting entries."""
from __future__ import annotations

from vaultlens.store import JsonStore, matches_saved


def test_matches_saved_ignores_order_and_empty_filters():
    saved = {"query": "rag", "filters": {"tags": ["a", "b"], "type": []}}
    assert matches_saved(saved, "rag", {"tags": ["b", "a"]})
    assert matches_saved({"query": "rag"}, "rag", {})


def test_matches_saved_detects_edits():
    saved = {"query": "rag", "filters": {"tags": ["a"]}}
    assert not matches_saved(saved, "rag2", {"tags": ["a"]})
    assert not matches_saved(saved, "rag", {"tags": ["a", "b"]})
    assert not matches_saved(saved, "rag", {})


def test_delete_search_roundtrip(tmp_path):
    store = JsonStore(str(tmp_path))
    store.save_search("one", {"query": "one"})
    store.save_search("two", {"query": "two"})
    store.delete_search("one")
    assert [s["name"] for s in store.saved_searches()] == ["two"]
    store.delete_search("missing")  # no error
    assert len(store.saved_searches()) == 1
