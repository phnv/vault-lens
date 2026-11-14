"""Custom LangChain `BaseRetriever` wrapping `HybridSearcher` (ADR-0009 D8 / section 6.8)."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from ..models import SearchSpec


class HybridRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by `HybridSearcher`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    searcher: Any
    spec: SearchSpec = Field(default_factory=SearchSpec)
    top_n: int = 8

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        resp = self.searcher.search(query, self.spec, top_n=self.top_n)
        return [
            Document(
                page_content=p.text,
                metadata={
                    "doc_id": p.doc_id,
                    "path": p.path,
                    "title": p.title,
                    "heading": p.heading,
                    "vault": p.vault,
                    "position": p.position,
                    "rrf_score": p.rrf_score,
                    "rerank_score": p.rerank_score,
                },
            )
            for p in resp.passages
        ]
