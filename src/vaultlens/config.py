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

    def vault(self, slug: str) -> VaultConfig:
        for v in self.vaults:
            if v.slug == slug:
                return v
        raise KeyError(f"unknown vault: {slug}")


def _load_vaults(path: Path) -> tuple[tuple[VaultConfig, ...], dict]:
    if not path.exists():
        return (), {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    vaults = []
    for item in raw.get("vaults", []):
        kwargs = {k: v for k, v in item.items() if k != "ignore"}
        if "ignore" in item:
            kwargs["ignore"] = tuple(item["ignore"])
        vaults.append(VaultConfig(**kwargs))
    return tuple(vaults), raw.get("index", {})


def _env_or(env: Mapping[str, str], key: str, fallback: Any) -> Any:
    val = env.get(key)
    if val not in (None, ""):
        return val
    return "off" if fallback is False else fallback


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    vaults, index_cfg = _load_vaults(Path(env.get("CONFIG_PATH", "config/vaultlens.yaml")))
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
        embedding_model=str(
            _env_or(
                env, "EMBEDDING_MODEL", index_cfg.get("embedding_model", Settings.embedding_model)
            )
        ),
        vaults=vaults,
        llm=llm,
    )
