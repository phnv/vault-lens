"""Filesystem walk (read-only) and hash diff against what is indexed."""
from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..config import VaultConfig


@dataclass(frozen=True)
class RawFile:
    path: str  # posix, relative to vault root
    text: str
    mtime: float


@dataclass(frozen=True)
class ScanDiff:
    new: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: int


def scan_vault(vault: VaultConfig) -> Iterator[RawFile]:
    """Yield every .md file under the vault root. Never writes, never follows symlinks."""
    root = Path(vault.path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"vault root not found: {root}")
    ignored = {p.strip("/") for p in vault.ignore}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and (f"{rel_dir}/{d}" if rel_dir else d) not in ignored
        )
        for name in sorted(filenames):
            if not name.lower().endswith(".md") or name.startswith("."):
                continue
            full = Path(dirpath) / name
            if full.is_symlink():
                continue
            rel = f"{rel_dir}/{name}" if rel_dir else name
            text = full.read_text(encoding="utf-8", errors="replace")
            yield RawFile(rel, text, full.stat().st_mtime)


def diff_hashes(local: dict[str, str], indexed: dict[str, str]) -> ScanDiff:
    """Both args map doc_id -> content_hash."""
    new = sorted(k for k in local if k not in indexed)
    changed = sorted(k for k in local if k in indexed and local[k] != indexed[k])
    removed = sorted(k for k in indexed if k not in local)
    return ScanDiff(new, changed, removed, unchanged=len(local) - len(new) - len(changed))
