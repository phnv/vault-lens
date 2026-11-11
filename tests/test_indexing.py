from vaultlens.indexing.service import resolve_duplicate_ids
from vaultlens.vault.parser import parse_note


def test_duplicate_frontmatter_ids_fall_back_to_path_ids():
    a = parse_note("v", "a.md", "---\nid: 1\n---\nx")
    b = parse_note("v", "b.md", "---\nid: 1\n---\ny")
    resolve_duplicate_ids([b, a])
    assert a.doc_id == "v:1" and b.doc_id == "v:path:b.md" and b.warnings


def test_settings_from_env_and_yaml(tmp_path=None):
    import tempfile
    from pathlib import Path

    from vaultlens.config import load_settings

    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "c.yaml"
        cfg.write_text("vaults:\n  - {slug: s, path: /v, ignore: [x]}\nindex: {alias: notes, chunk_max_chars: 900}\n")
        s = load_settings({
            "CONFIG_PATH": str(cfg),
            "ES_URL": "http://h:9200",
            "GEMINI_API_KEY": "SECRET-KEY-123",
            "GROQ_API_KEY": "GROQ-SECRET-456",
            "GROQ_MODEL": "qwen/qwen3.8-27b",
            "LLM_DEFAULT_PROVIDER": "ollama",
        })
    assert s.vault("s").ignore == ("x",) and s.chunk_max_chars == 900
    assert s.es_url == "http://h:9200" and s.llm.default_provider == "ollama"
    assert s.llm.groq_model == "qwen/qwen3.8-27b"
    assert "SECRET-KEY-123" not in repr(s) and "GROQ-SECRET-456" not in repr(s)
