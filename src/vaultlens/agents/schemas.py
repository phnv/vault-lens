"""Pydantic schemas for structured LLM outputs in Tier b and Tier c (ADR-0009 section 6.8–6.10)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    """Tier b query transformation output."""

    rewrites: list[str] = Field(
        default_factory=list,
        description="Up to 3 alternative phrasings (including PT<->EN translation when helpful).",
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Up to 4 atomic sub-questions when multi-part; empty if atomic.",
    )


class AgentPlan(BaseModel):
    """Tier c initial plan: query expansion + optional facet filter proposal."""

    rewrites: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    filters: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Optional facet filters (type, status, tags, folder) from allowed values.",
    )
    rationale: str = Field(
        default="",
        description="Short 1-sentence rationale for the query plan and any proposed filters.",
    )


class Grade(BaseModel):
    """Tier c evidence grading output."""

    sufficient: bool = Field(
        description="True if passages answer the question (or the vault clearly lacks more)."
    )
    missing: list[str] = Field(
        default_factory=list,
        description="Specific sub-questions or facts still unanswered by the current evidence.",
    )
    next_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 refined search queries to try next when sufficient is False.",
    )
