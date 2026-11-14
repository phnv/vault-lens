"""Cross-encoder reranker (ADR-0009 D1/D3/R1). Lazy, cached, CPU-only."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from ..config import RerankConfig, Settings


class Reranker(Protocol):
    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]: ...


_TAG_RE = re.compile(r"</?mark>")


def strip_highlight_tags(snippet: str) -> str:
    return _TAG_RE.sub("", snippet)


def prepare_rerank_text(
    query: str,
    heading: str,
    text: str,
    snippet: str = "",
    *,
    max_tokens: int = 512,
    oversized_chars: int = 1600,
) -> str:
    """Build passage text for the cross-encoder (ADR-0009 R1).

    - Full chunk text (`"{heading}\\n{text}"`) is used when `len(text) <= oversized_chars`.
    - For oversized chunks (`len(text) > oversized_chars`), fall back to the ES highlight snippet
      (with `<mark>` tags stripped) when present (lexical leg), else head truncation.
    - Truncate to a conservative character budget derived from `max_tokens` minus query tokens.
    """
    q_tokens = max(1, len(query.split()))
    budget_tokens = max(64, max_tokens - q_tokens - 4)
    char_budget = budget_tokens * 4

    body = text
    if len(text) > oversized_chars:
        clean_snip = strip_highlight_tags(snippet).strip()
        body = clean_snip if clean_snip else text[:oversized_chars]

    combined = f"{heading.strip()}\n{body}" if heading.strip() else body
    return combined[:char_budget]


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig, model_factory=None) -> None:
        self.cfg = cfg
        self._factory = model_factory
        self._model = None

    def _load(self):
        if self._model is None:
            if self._factory is not None:
                self._model = self._factory(self.cfg.model)
            else:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.cfg.model, max_length=512)
        return self._model

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        scores = self._load().predict(pairs)
        return [float(s) for s in scores]


@lru_cache(maxsize=4)
def _cached(cfg: RerankConfig) -> Reranker:
    return CrossEncoderReranker(cfg)


def get_reranker(settings: Settings) -> Reranker | None:
    """Cached singleton per config; returns None when reranking is disabled."""
    if not settings.rerank.enabled:
        return None
    return _cached(settings.rerank)
