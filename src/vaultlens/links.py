"""Deep links back to the original file."""

from __future__ import annotations

from urllib.parse import quote

from .config import VaultConfig


def obsidian_uri(vault: VaultConfig, rel_path: str) -> str | None:
    if not vault.obsidian_vault_name:
        return None
    file = rel_path[:-3] if rel_path.lower().endswith(".md") else rel_path
    return f"obsidian://open?vault={quote(vault.obsidian_vault_name)}&file={quote(file)}"


def editor_uri(vault: VaultConfig, rel_path: str) -> str | None:
    if not vault.host_path:
        return None
    return f"vscode://file/{quote(vault.host_path.rstrip('/') + '/' + rel_path)}"
