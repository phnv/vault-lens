"""Loads the versioned index definition shipped inside the package."""
from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any


MAPPING_VERSION = 1  # bump when adding search/mappings/notes_v<N>.json (a schema change)


def load_mapping(version: int = MAPPING_VERSION) -> dict[str, Any]:
    ref = resources.files("vaultlens.search").joinpath("mappings", f"notes_v{version}.json")
    return json.loads(ref.read_text(encoding="utf-8"))


def index_name(alias: str, version: int) -> str:
    return f"{alias}_v{version}"


def version_of(index: str) -> int:
    m = re.search(r"_v(\d+)$", index)
    return int(m.group(1)) if m else 0


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
