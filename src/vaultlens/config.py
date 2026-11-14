"""Settings from environment variables plus the vault registry in YAML."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VaultConfig:
    slug: str
    path: str  # path INSIDE the container
    label: str = ""
    obsidian_vault_name: str | None = None
    host_path: str | None = None  # path on the host, only for "open in editor" links
    ignore: tuple[str, ...] = (".obsidian", ".trash", "_Meta/templates")


@dataclass(frozen=True)
class LLMConfig:
    default_provider: str = "off"  # off | ollama | gemini | groq
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen2.5:1.5b-instruct"
    gemini_api_key: str = field(default="", repr=False)  # never log or display
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str = field(default="", repr=False)  # never log or display
    groq_model: str = "openai/gpt-oss-120b"
    timeout_s: float = 120.0


@dataclass(frozen=True)
class EmbedConfig:
    provider: str = "sentence_transformers"  # sentence_transformers | ollama
    model: str = "BAAI/bge-m3"
    dims: int = 1024  # must equal the mapping dims
    max_tokens: int = 512
    batch: int = 16


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    mode: str = "blend"  # blend | replace
    pool: int = 20


@dataclass(frozen=True)
class HybridConfig:
    candidates: int = 30  # depth of each leg; kNN num_candidates = 3x
    rrf_k: int = 60


@dataclass(frozen=True)
class AgentConfig:
    max_iterations: int = 3
    max_llm_calls: int = 8
    timeout_s: float = 120.0
    min_hits: int = 3


@dataclass(frozen=True)
class Settings:
    es_url: str = "http://elasticsearch:9200"
    es_alias: str = "notes"
    data_dir: str = "/data"
    parser: str = "default"
    chunker: str = "heading"
    chunk_max_chars: int = 1500
    chunk_overlap: int = 0
    embedding_model: str = "off"
    vaults: tuple[VaultConfig, ...] = ()
    llm: LLMConfig = LLMConfig()
    embed: EmbedConfig = EmbedConfig()
    rerank: RerankConfig = RerankConfig()
    hybrid: HybridConfig = HybridConfig()
    agent: AgentConfig = AgentConfig()

    @property
    def embeddings_enabled(self) -> bool:
        return self.embedding_model != "off"

    def vault(self, slug: str) -> VaultConfig:
        for v in self.vaults:
            if v.slug == slug:
                return v
        raise KeyError(f"unknown vault: {slug}")


def _load_vaults(path: Path) -> tuple[tuple[VaultConfig, ...], dict[str, Any]]:
    if not path.exists():
        return (), {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    vaults = []
    for item in raw.get("vaults", []):
        kwargs = {k: v for k, v in item.items() if k != "ignore"}
        if "ignore" in item:
            kwargs["ignore"] = tuple(item["ignore"])
        vaults.append(VaultConfig(**kwargs))
    return tuple(vaults), raw


def _env_or(env: Mapping[str, str], key: str, fallback: Any) -> Any:
    val = env.get(key)
    if val not in (None, ""):
        return val
    return "off" if fallback is False else fallback


def _bool(val: Any) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _semantic_configs(
    env: Mapping[str, str], raw: Mapping[str, Any], embedding_model: str
) -> tuple[EmbedConfig, RerankConfig, HybridConfig, AgentConfig]:
    """env beats yaml (`semantic:`) beats defaults; `embedding_model` comes from `index:`."""

    def pick(section: str, key: str, env_key: str, default: Any) -> Any:
        return _env_or(env, env_key, (raw.get(section) or {}).get(key, default))

    e, r, h, a = EmbedConfig, RerankConfig, HybridConfig, AgentConfig
    embed = EmbedConfig(
        provider=str(pick("embed", "provider", "EMBED_PROVIDER", e.provider)),
        model=e.model if embedding_model == "off" else embedding_model,
        dims=int(pick("embed", "dims", "EMBED_DIMS", e.dims)),
        max_tokens=int(pick("embed", "max_tokens", "EMBED_MAX_TOKENS", e.max_tokens)),
        batch=int(pick("embed", "batch", "EMBED_BATCH", e.batch)),
    )
    rerank = RerankConfig(
        enabled=_bool(pick("rerank", "enabled", "RERANK_ENABLED", r.enabled)),
        model=str(pick("rerank", "model", "RERANK_MODEL", r.model)),
        mode=str(pick("rerank", "mode", "RERANK_MODE", r.mode)),
        pool=int(pick("rerank", "pool", "RERANK_POOL", r.pool)),
    )
    hybrid = HybridConfig(
        candidates=int(pick("hybrid", "candidates", "HYBRID_CANDIDATES", h.candidates)),
        rrf_k=int(pick("hybrid", "rrf_k", "RRF_K", h.rrf_k)),
    )
    agent = AgentConfig(
        max_iterations=int(
            pick("agent", "max_iterations", "AGENT_MAX_ITERATIONS", a.max_iterations)
        ),
        max_llm_calls=int(pick("agent", "max_llm_calls", "AGENT_MAX_LLM_CALLS", a.max_llm_calls)),
        timeout_s=float(pick("agent", "timeout_s", "AGENT_TIMEOUT_S", a.timeout_s)),
        min_hits=int(pick("agent", "min_hits", "AGENT_MIN_HITS", a.min_hits)),
    )
    if rerank.mode not in ("blend", "replace"):
        raise ValueError(f"RERANK_MODE must be 'blend' or 'replace', got {rerank.mode!r}")
    if embed.provider not in ("sentence_transformers", "ollama"):
        raise ValueError(
            f"EMBED_PROVIDER must be sentence_transformers or ollama: {embed.provider!r}"
        )
    return embed, rerank, hybrid, agent


def _normalize_embedding_model(value: Any) -> str:
    """'off' (or empty/false/0) disables vectors; any other value is the exact model ID."""
    text = str(value).strip()
    return "off" if text.lower() in ("", "off", "false", "0", "none") else text


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    vaults, raw_cfg = _load_vaults(Path(env.get("CONFIG_PATH", "config/vaultlens.yaml")))
    index_cfg = raw_cfg.get("index", {}) or {}
    embedding_model = _normalize_embedding_model(
        index_cfg.get("embedding_model", Settings.embedding_model)
    )
    embed, rerank, hybrid, agent = _semantic_configs(
        env, raw_cfg.get("semantic", {}) or {}, embedding_model
    )
    llm = LLMConfig(
        default_provider=env.get("LLM_PROVIDER", env.get("LLM_DEFAULT_PROVIDER", "off")),
        ollama_url=env.get("OLLAMA_BASE_URL", env.get("OLLAMA_URL", LLMConfig.ollama_url)),
        ollama_model=env.get("OLLAMA_MODEL", LLMConfig.ollama_model),
        gemini_api_key=env.get("GEMINI_API_KEY", ""),
        gemini_model=env.get("GEMINI_MODEL", LLMConfig.gemini_model),
        groq_api_key=env.get("GROQ_API_KEY", ""),
        groq_model=env.get("GROQ_MODEL", LLMConfig.groq_model),
        timeout_s=float(env.get("LLM_TIMEOUT", env.get("LLM_TIMEOUT_S", LLMConfig.timeout_s))),
    )
    return Settings(
        es_url=_env_or(env, "ES_URL", Settings.es_url),
        es_alias=_env_or(env, "ES_ALIAS", index_cfg.get("alias", Settings.es_alias)),
        data_dir=_env_or(env, "DATA_DIR", Settings.data_dir),
        parser=str(_env_or(env, "PARSER", index_cfg.get("parser", Settings.parser))),
        chunker=str(_env_or(env, "CHUNKER", index_cfg.get("chunker", Settings.chunker))),
        chunk_max_chars=int(
            _env_or(
                env, "CHUNK_MAX_CHARS", index_cfg.get("chunk_max_chars", Settings.chunk_max_chars)
            )
        ),
        chunk_overlap=int(
            _env_or(env, "CHUNK_OVERLAP", index_cfg.get("chunk_overlap", Settings.chunk_overlap))
        ),
        embedding_model=embedding_model,
        vaults=vaults,
        llm=llm,
        embed=embed,
        rerank=rerank,
        hybrid=hybrid,
        agent=agent,
    )
