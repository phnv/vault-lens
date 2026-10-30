"""Settings from environment variables plus the vault registry in YAML."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

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
    default_provider: str = "off"  # off | ollama | gemini
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen2.5:1.5b-instruct"
    gemini_api_key: str = field(default="", repr=False)  # never log or display
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    timeout_s: float = 120.0


@dataclass(frozen=True)
class Settings:
    es_url: str = "http://elasticsearch:9200"
    es_alias: str = "notes"
    data_dir: str = "/data"
    chunk_max_chars: int = 1500
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


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    vaults, index_cfg = _load_vaults(Path(env.get("CONFIG_PATH", "config/vaultlens.yaml")))
    llm = LLMConfig(
        default_provider=env.get("LLM_PROVIDER", env.get("LLM_DEFAULT_PROVIDER", "off")),
        ollama_url=env.get("OLLAMA_BASE_URL", env.get("OLLAMA_URL", LLMConfig.ollama_url)),
        ollama_model=env.get("OLLAMA_MODEL", LLMConfig.ollama_model),
        gemini_api_key=env.get("GEMINI_API_KEY", ""),
        gemini_model=env.get("GEMINI_MODEL", LLMConfig.gemini_model),
        timeout_s=float(env.get("LLM_TIMEOUT", env.get("LLM_TIMEOUT_S", LLMConfig.timeout_s))),
    )
    return Settings(
        es_url=env.get("ES_URL", Settings.es_url),
        es_alias=env.get("ES_ALIAS", index_cfg.get("alias", Settings.es_alias)),
        data_dir=env.get("DATA_DIR", Settings.data_dir),
        chunk_max_chars=int(index_cfg.get("chunk_max_chars", Settings.chunk_max_chars)),
        vaults=vaults,
        llm=llm,
    )
