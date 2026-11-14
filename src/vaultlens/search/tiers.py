"""Search tier registry (ADR-0009 section 6.7). Stdlib only; not yet wired into the UI (M0)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    id: str
    label: str
    needs_llm: bool
    needs_embeddings: bool
    caption: str


TIERS: tuple[Tier, ...] = (
    Tier("simple", "Simple Search", False, False, "BM25, no LLM"),
    Tier("ask", "Ask AI", True, False, "BM25 + 1 LLM call"),
    Tier("hybrid", "Hybrid", False, True, "BM25 + vectors + rerank, no LLM"),
    Tier("ai_search", "AI Search", True, True, "Query expansion + answer, 2 LLM calls"),
    Tier("agentic", "Agentic", True, True, "Self-correcting loop, up to 8 LLM calls"),
)

TIERS_BY_ID: dict[str, Tier] = {t.id: t for t in TIERS}


def get_tier(tier_id: str) -> Tier:
    try:
        return TIERS_BY_ID[tier_id]
    except KeyError:
        raise KeyError(f"unknown tier: {tier_id}") from None


def evaluate_tier_status(
    tier: Tier,
    *,
    llm_provider: str,
    llm_error: str | None = None,
    config_embedding_model: str = "off",
    live_embedding_model: str = "off",
    total_chunks: int | None = None,
    embedded_chunks: int | None = None,
) -> tuple[bool, str]:
    """Pure availability check for `tier` (ADR-0009 section 6.7 / R2)."""
    if tier.needs_embeddings:
        if config_embedding_model == "off":
            return (
                False,
                "Embeddings are disabled (index.embedding_model: off in config/vaultlens.yaml). "
                "Enable embeddings and run Reindex.",
            )
        if live_embedding_model == "off":
            return (
                False,
                "The live index has no embeddings. Run Full reindex (`make reindex`).",
            )
        if live_embedding_model != config_embedding_model:
            return (
                False,
                f"Index embedding model ({live_embedding_model!r}) does not match configured "
                f"model ({config_embedding_model!r}). Run Full reindex (`make reindex`).",
            )
        if (
            total_chunks is not None
            and embedded_chunks is not None
            and total_chunks > 0
            and embedded_chunks == 0
        ):
            return (
                False,
                f"The live index has {total_chunks:,} chunks but 0 embedded vectors. "
                "Run Full reindex (`make reindex`).",
            )
    if tier.needs_llm:
        if llm_provider == "off":
            return (
                False,
                f"Choose an LLM provider in the sidebar to use {tier.label}.",
            )
        if tier.id in ("ai_search", "agentic") and llm_provider not in ("groq", "gemini"):
            return (
                False,
                f"{tier.label} requires Groq or Gemini (selected provider: {llm_provider!r}).",
            )
        if llm_error:
            return False, llm_error
    return True, "ok"
