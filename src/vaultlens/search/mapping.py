"""Loads the versioned index definition shipped inside the package."""
from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

MAPPING_VERSION = 1  # bump when adding search/mappings/notes_v<N>.json (a schema change)


def load_mapping(version: int = MAPPING_VERSION, embedding_model: str = "off") -> dict[str, Any]:
    ref = resources.files("vaultlens.search").joinpath("mappings", f"notes_v{version}.json")
    data = json.loads(ref.read_text(encoding="utf-8"))
    meta = dict(data.get("mappings", {}).get("_meta", {}))
    meta["mapping_version"] = version
    meta["embedding_model"] = embedding_model
    data.setdefault("mappings", {})["_meta"] = meta
    return data


def index_name(alias: str, version: int) -> str:
    return f"{alias}_v{version}"


def version_of(index: str) -> int:
    m = re.search(r"_v(\d+)$", index)
    return int(m.group(1)) if m else 0


def resolve_index_meta(index: str | None, raw_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Return effective index _meta, defaulting missing fields for pre-_meta indices."""
    if index is None:
        return {}
    meta = dict(raw_meta or {})
    meta.setdefault("mapping_version", version_of(index) or MAPPING_VERSION)
    meta.setdefault("embedding_model", "off")
    return meta


def check_meta_compatibility(
    index: str | None,
    raw_meta: dict[str, Any] | None,
    *,
    mapping_version: int = MAPPING_VERSION,
    embedding_model: str = "off",
) -> tuple[bool, str]:
    """Check whether incremental sync is safe against the live index's _meta."""
    if index is None:
        return True, "no index yet"
    meta = resolve_index_meta(index, raw_meta)
    live_ver = int(meta["mapping_version"])
    live_emb = str(meta["embedding_model"])
    reasons: list[str] = []
    if live_ver != mapping_version:
        reasons.append(f"mapping_version (index={live_ver}, code={mapping_version})")
    if live_emb != embedding_model:
        reasons.append(f"embedding_model (index={live_emb!r}, config={embedding_model!r})")
    if reasons:
        return (
            False,
            f"Index {index!r} is incompatible with current settings ({', '.join(reasons)}). "
            "Run a Full reindex (`make reindex` or `vaultlens reindex`) "
            "to build the next index and swap the alias.",
        )
    return True, "ok"


def mapped_fields(mapping: dict[str, Any]) -> dict[str, str]:
    """Flatten mapping properties (incl. multi-fields) to {dotted.path: type}."""
    out: dict[str, str] = {}

    def walk(props: dict[str, Any], prefix: str) -> None:
        for name, spec in props.items():
            path = f"{prefix}{name}"
            out[path] = spec.get("type", "object")
            walk(spec.get("properties", {}), f"{path}.")
            walk(spec.get("fields", {}), f"{path}.")

    walk(mapping["mappings"]["properties"], "")
    return out
