"""`make pull-models`: download the embedding and rerank models into HF_HOME (multi-GB: GATE)."""

from __future__ import annotations

from ..config import load_settings


def main() -> None:
    s = load_settings()
    if s.embed.provider == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        print(f"fetching embedder {s.embed.model} (sentence_transformers) ...")
        SentenceTransformer(s.embed.model)
    elif s.embed.provider == "fastembed":
        from fastembed import TextEmbedding

        print(f"fetching embedder {s.embed.model} (fastembed) ...")
        TextEmbedding(model_name=s.embed.model)
    if s.rerank.enabled:
        from sentence_transformers import CrossEncoder

        print(f"fetching reranker {s.rerank.model} ...")
        CrossEncoder(s.rerank.model)
    print("done")


if __name__ == "__main__":
    main()
