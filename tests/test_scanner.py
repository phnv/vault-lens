from pathlib import Path

from vaultlens.config import VaultConfig
from vaultlens.vault.scanner import diff_hashes, scan_vault

FIX = Path(__file__).parent / "fixtures" / "vault"


def test_scan_ignores_hidden_and_configured_folders():
    vault = VaultConfig(slug="fixture", path=str(FIX))
    paths = [f.path for f in scan_vault(vault)]
    assert paths == [
        "00-Inbox/ideia-solta.md",
        "10-Projects/dayo/dayo-notes-server.md",
        "30-Resources/search/Elasticsearch analyzers.md",
    ]


def test_scan_missing_root_raises():
    vault = VaultConfig(slug="x", path="/definitely/not/here")
    try:
        list(scan_vault(vault))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_diff_hashes():
    d = diff_hashes({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "X", "z": "9"})
    assert (d.new, d.changed, d.removed, d.unchanged) == (["c"], ["b"], ["z"], 1)
