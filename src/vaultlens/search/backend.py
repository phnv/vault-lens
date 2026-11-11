"""Thin Elasticsearch wrapper. The ONLY module (with indexing/) that imports `elasticsearch`.

Written against elasticsearch-py 8.x. UNVERIFIED against a live server: first integration task in
docs/ROADMAP.md (M1).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..models import SearchSpec
from .mapping import (
    check_meta_compatibility,
    index_name,
    load_mapping,
    resolve_index_meta,
    version_of,
)
from .query_builder import build_search_body


class SearchBackend:
    def __init__(self, url: str, alias: str = "notes") -> None:
        from elasticsearch import Elasticsearch  # lazy: keeps the core importable without it

        self.es = Elasticsearch(url, request_timeout=30)
        self.alias = alias

    # health / info ---------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        try:
            info = self.es.cluster.health(wait_for_status="yellow", timeout="3s")
            return True, str(info["status"])
        except Exception as exc:  # connection refused, timeout...
            return False, type(exc).__name__

    def current_index(self) -> str | None:
        if not self.es.indices.exists_alias(name=self.alias):
            return None
        return next(iter(self.es.indices.get_alias(name=self.alias)), None)

    def index_meta(self, index: str | None = None) -> dict[str, Any]:
        target = index or self.current_index()
        if target is None:
            return {}
        try:
            resp = self.es.indices.get_mapping(index=target)
            raw_meta = resp.get(target, {}).get("mappings", {}).get("_meta")
        except Exception:
            raw_meta = None
        return resolve_index_meta(target, raw_meta)

    def check_index_compatibility(self, embedding_model: str = "off") -> tuple[bool, str]:
        current = self.current_index()
        if current is None:
            return True, "no index yet"
        return check_meta_compatibility(
            current, self.index_meta(current), embedding_model=embedding_model
        )

    def stats(self) -> dict[str, Any]:
        index = self.current_index()
        if index is None:
            return {"index": None, "docs": 0, "chunks": 0, "size_bytes": 0, "meta": {}}
        resp = self.es.search(
            index=self.alias,
            body={
                "size": 0,
                "aggs": {
                    "chunks": {"nested": {"path": "chunks"}},
                    "vaults": {"terms": {"field": "vault", "size": 50}},
                },
            },
        )
        st = self.es.indices.stats(index=index, metric="store")
        buckets = resp["aggregations"]["vaults"]["buckets"]
        return {
            "index": index,
            "docs": resp["hits"]["total"]["value"],
            "chunks": resp["aggregations"]["chunks"]["doc_count"],
            "per_vault": {b["key"]: b["doc_count"] for b in buckets},
            "size_bytes": st["_all"]["primaries"]["store"]["size_in_bytes"],
            "meta": self.index_meta(index),
        }

    # index lifecycle ---------------------------------------------------------
    def ensure_index(self, embedding_model: str = "off") -> str:
        current = self.current_index()
        if current:
            return current
        name = index_name(self.alias, 1)
        self.es.indices.create(index=name, body=load_mapping(embedding_model=embedding_model))
        self.es.indices.put_alias(index=name, name=self.alias)
        return name

    def create_next_index(self, embedding_model: str = "off") -> str:
        current = self.current_index()
        name = index_name(self.alias, version_of(current) + 1 if current else 1)
        if name != current:
            self.es.indices.delete(index=name, ignore_unavailable=True)  # leftover of a failed run
        self.es.indices.create(index=name, body=load_mapping(embedding_model=embedding_model))
        return name

    def swap_alias(self, old: str | None, new: str) -> None:
        actions: list[dict[str, Any]] = [{"add": {"index": new, "alias": self.alias}}]
        if old:
            actions.append({"remove": {"index": old, "alias": self.alias}})
        self.es.indices.update_aliases(actions=actions)
        if old:
            self.es.indices.delete(index=old)

    def refresh(self, index: str | None = None) -> None:
        self.es.indices.refresh(index=index or self.alias)

    # documents ---------------------------------------------------------------
    def bulk(self, actions: Iterable[dict[str, Any]]) -> tuple[int, list[Any]]:
        from elasticsearch import helpers

        ok, errors = helpers.bulk(self.es, actions, raise_on_error=False, stats_only=False)
        return ok, list(errors) if isinstance(errors, list) else []

    def indexed_hashes(self, vault: str) -> dict[str, str]:
        from elasticsearch import helpers

        if self.current_index() is None:
            return {}
        hits = helpers.scan(
            self.es,
            index=self.alias,
            query={"query": {"term": {"vault": vault}}, "_source": ["content_hash"]},
        )
        return {h["_id"]: h["_source"]["content_hash"] for h in hits}

    # search ------------------------------------------------------------------
    def search(self, spec: SearchSpec) -> dict[str, Any]:
        return self.raw_search(build_search_body(spec))

    def raw_search(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.es.search(index=self.alias, body=body).body

    def analyze(self, analyzer: str, text: str) -> list[str]:
        resp = self.es.indices.analyze(
            index=self.current_index(), body={"analyzer": analyzer, "text": text}
        )
        return [t["token"] for t in resp["tokens"]]

    def rank_eval(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.es.rank_eval(index=self.alias, body=body).body
