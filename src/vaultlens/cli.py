"""Headless entrypoint. Lets humans and coding agents verify the core without the UI.

  vaultlens status | sync [--vault SLUG] | reindex | search QUERY | ask QUESTION | eval
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings


def _backend(settings):
    from .search.backend import SearchBackend

    return SearchBackend(settings.es_url, settings.es_alias)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vaultlens")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("sync")
    s.add_argument("--vault")
    sub.add_parser("reindex")
    q = sub.add_parser("search")
    q.add_argument("query")
    q.add_argument("--vault", action="append", default=[])
    q.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE")
    q.add_argument("--size", type=int, default=10)
    q.add_argument("--show-query", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--provider", choices=["ollama", "gemini", "groq"], default="ollama")
    a.add_argument("--model", default=None)
    e = sub.add_parser("eval")
    e.add_argument("--judgments", default="config/judged_queries.yaml")
    e.add_argument("--strategies", nargs="+", default=["A", "B", "C"])
    args = p.parse_args(argv)

    settings = load_settings()
    backend = _backend(settings)

    if args.cmd == "status":
        ok, state = backend.health()
        print(json.dumps({"elasticsearch": {"ok": ok, "state": state}, **(backend.stats() if ok else {})}, indent=2))
        return 0 if ok else 1

    if args.cmd in ("sync", "reindex"):
        from .indexing.service import IndexService
        from .store import JsonStore

        svc = IndexService(backend, settings, JsonStore(settings.data_dir))
        if args.cmd == "reindex":
            reports = svc.full_reindex()
        else:
            slugs = [args.vault] if args.vault else [v.slug for v in settings.vaults]
            reports = [svc.sync(slug) for slug in slugs]
        print(json.dumps([r.to_dict() for r in reports], indent=2))
        return 0

    if args.cmd == "search":
        from .models import SearchSpec
        from .search.query_builder import compiled_query
        from .search.results import parse_response

        filters: dict[str, list[str]] = {}
        for item in args.filter:
            field, _, value = item.partition("=")
            filters.setdefault(field, []).append(value)
        spec = SearchSpec(query=args.query, vaults=args.vault, filters=filters, size=args.size, facets=False)
        if args.show_query:
            print(compiled_query(spec))
        resp = parse_response(backend.search(spec))
        print(f"{resp.total} results in {resp.took_ms} ms")
        for h in resp.hits:
            print(f"- {h.title}  [{h.vault}:{h.path}]  score={h.score}")
        return 0

    if args.cmd == "ask":
        from .ask import AskService
        from .llm.base import LLMError
        from .llm.factory import get_client

        try:
            llm = get_client(args.provider, settings.llm, groq_model=args.model)
            result = AskService(backend.search, llm).answer(args.question)  # type: ignore[arg-type]
        except LLMError as exc:
            print(f"LLM error: {exc}", file=sys.stderr)
            return 2
        print(result.answer)
        for src in result.sources:
            print(f"  [{src.n}] {src.title} ({src.path})")
        return 0

    if args.cmd == "eval":
        from .evaluate import load_judgments, run_evaluation

        judged = load_judgments(args.judgments)
        if not judged:
            print("no judged queries yet; see config/judged_queries.yaml", file=sys.stderr)
            return 1
        for r in run_evaluation(backend, judged, args.strategies):
            print(f"{r.strategy} ({r.label}): nDCG@10 mean={r.mean:.3f}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
