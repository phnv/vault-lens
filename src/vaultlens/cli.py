"""Headless entrypoint. Lets humans and coding agents verify the core without the UI.

vaultlens status | sync [--vault SLUG] [--force] | reindex | search QUERY [--mode simple|hybrid]
vaultlens ask QUESTION [--tier ask|ai_search|agentic] [--provider ollama|gemini|groq] [--trace]
vaultlens eval
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
    s.add_argument(
        "--force",
        action="store_true",
        help="Re-index all notes even when content hashes match",
    )
    sub.add_parser("reindex")
    q = sub.add_parser("search")
    q.add_argument("query")
    q.add_argument("--mode", choices=["simple", "hybrid"], default="simple")
    q.add_argument("--vault", action="append", default=[])
    q.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE")
    q.add_argument("--size", type=int, default=10)
    q.add_argument("--show-query", action="store_true")
    q.add_argument("--trace", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--tier", choices=["ask", "ai_search", "agentic"], default="ask")
    a.add_argument("--provider", choices=["ollama", "gemini", "groq"], default=None)
    a.add_argument("--model", default=None)
    a.add_argument("--trace", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--judgments", default="config/judged_queries.yaml")
    e.add_argument("--strategies", nargs="+", default=["A", "B", "C"])
    args = p.parse_args(argv)

    settings = load_settings()
    backend = _backend(settings)

    if args.cmd == "status":
        from .search.mapping import MAPPING_VERSION
        from .vault.parser import CHUNKER_VERSION, PARSER_VERSION

        ok, state = backend.health()
        compat_ok, compat_msg = (
            backend.check_index_compatibility(settings.embedding_model)
            if ok
            else (False, "unreachable")
        )
        payload = {
            "elasticsearch": {"ok": ok, "state": state},
            "pipeline": {
                "parser": settings.parser,
                "parser_version": PARSER_VERSION,
                "chunker": settings.chunker,
                "chunker_version": CHUNKER_VERSION,
                "chunk_max_chars": settings.chunk_max_chars,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "mapping_version": MAPPING_VERSION,
            },
            "index_compatible": {"ok": compat_ok, "message": compat_msg},
            **(backend.stats() if ok else {}),
        }
        print(json.dumps(payload, indent=2))
        return 0 if ok else 1

    if args.cmd in ("sync", "reindex"):
        from .indexing.service import IndexCompatibilityError, IndexService
        from .semantic.embedder import get_embedder
        from .store import JsonStore

        emb = get_embedder(settings) if settings.embeddings_enabled else None
        svc = IndexService(backend, settings, JsonStore(settings.data_dir), embedder=emb)
        try:
            if args.cmd == "reindex":
                reports = svc.full_reindex()
            else:
                slugs = [args.vault] if args.vault else [v.slug for v in settings.vaults]
                reports = [svc.sync(slug, force=args.force) for slug in slugs]
        except IndexCompatibilityError as exc:
            print(f"Index compatibility error: {exc}", file=sys.stderr)
            return 2
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
        spec = SearchSpec(
            query=args.query, vaults=args.vault, filters=filters, size=args.size, facets=False
        )
        if args.show_query:
            print(compiled_query(spec))

        if args.mode == "hybrid":
            from .agents.trace import TraceSink
            from .search.hybrid import HybridSearcher
            from .semantic.embedder import get_embedder
            from .semantic.reranker import get_reranker

            sink = TraceSink()
            searcher = HybridSearcher(
                backend.raw_search,
                get_embedder(settings),
                get_reranker(settings),
                cfg=settings.hybrid,
                rerank_cfg=settings.rerank,
                trace=sink,
                oversized_chars=max(1600, settings.chunk_max_chars),
            )
            hresp = searcher.search(args.query, spec, top_n=args.size)
            if args.trace and sink.events:
                print(sink.to_jsonl())
            print(f"{hresp.total} results in {hresp.took_ms} ms")
            for h in hresp.hits:
                print(f"- {h.title}  [{h.vault}:{h.path}]  score={h.score}")
            return 0

        resp = parse_response(backend.search(spec))
        print(f"{resp.total} results in {resp.took_ms} ms")
        for h in resp.hits:
            print(f"- {h.title}  [{h.vault}:{h.path}]  score={h.score}")
        return 0

    if args.cmd == "ask":
        from .ask import AskService
        from .llm.base import LLMError
        from .llm.factory import get_client
        from .models import SearchSpec

        provider = args.provider or settings.llm.default_provider
        if args.tier == "ask":
            try:
                llm = get_client(provider, settings.llm, groq_model=args.model)
                if llm is None:
                    raise LLMError("LLM provider is 'off'; pass --provider ollama|gemini|groq.")
                result = AskService(backend.search, llm).answer(args.question)
            except LLMError as exc:
                print(f"LLM error: {exc}", file=sys.stderr)
                return 2
            print(result.answer)
            for src in result.sources:
                print(f"  [{src.n}] {src.title} ({src.path})")
            return 0

        from .agents.agentic import run_agentic_search
        from .agents.ai_search import run_ai_search
        from .agents.llm import get_chat_model
        from .agents.trace import TraceSink
        from .search.hybrid import HybridSearcher
        from .semantic.embedder import get_embedder
        from .semantic.reranker import get_reranker

        try:
            chat_model = get_chat_model(provider, settings.llm, groq_model=args.model)
            sink = TraceSink()
            searcher = HybridSearcher(
                backend.raw_search,
                get_embedder(settings),
                get_reranker(settings),
                cfg=settings.hybrid,
                rerank_cfg=settings.rerank,
                trace=sink,
                oversized_chars=max(1600, settings.chunk_max_chars),
            )
            spec = SearchSpec(query=args.question, facets=False)
            if args.tier == "ai_search":
                out = run_ai_search(args.question, spec, chat_model, searcher, sink=sink)
            else:
                out = run_agentic_search(
                    args.question, spec, chat_model, searcher, cfg=settings.agent, sink=sink
                )
        except LLMError as exc:
            print(f"LLM error: {exc}", file=sys.stderr)
            return 2

        if args.trace and sink.events:
            print(sink.to_jsonl())
        if out.answer:
            print(out.answer)
        for src in out.sources:
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
