"""Tiny JSON state in DATA_DIR: sync reports and saved searches. Atomic writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def matches_saved(saved_spec: dict[str, Any], query: str, filters: dict[str, list[str]]) -> bool:
    """True when the current query and filters equal a saved spec (filter order and empty lists ignored)."""

    def norm(f: dict[str, list[str]] | None) -> dict[str, set[str]]:
        return {k: set(v) for k, v in (f or {}).items() if v}

    return saved_spec.get("query", "") == query and norm(saved_spec.get("filters")) == norm(filters)


class JsonStore:
    def __init__(self, data_dir: str) -> None:
        self.dir = Path(data_dir)

    def _read(self, name: str, default: Any) -> Any:
        p = self.dir / name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def _write(self, name: str, data: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.dir / name)

    # sync state -----------------------------------------------------------
    def sync_state(self) -> dict[str, dict[str, Any]]:
        return self._read("sync_state.json", {})

    def record_sync(self, vault: str, report: dict[str, Any]) -> None:
        state = self.sync_state()
        state[vault] = report
        self._write("sync_state.json", state)

    # saved searches -------------------------------------------------------
    def saved_searches(self) -> list[dict[str, Any]]:
        return self._read("saved_searches.json", [])

    def save_search(self, name: str, spec: dict[str, Any]) -> None:
        items = [s for s in self.saved_searches() if s["name"] != name]
        items.append({"name": name, "spec": spec})
        self._write("saved_searches.json", items)

    def delete_search(self, name: str) -> None:
        self._write("saved_searches.json", [s for s in self.saved_searches() if s["name"] != name])
