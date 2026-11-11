from pathlib import Path

import pytest

import vaultlens.vault.parser as parser_mod
from vaultlens.models import Chunk
from vaultlens.vault.parser import (
    CHUNKERS,
    chunk_markdown,
    extract_links,
    get_chunker,
    get_parser,
    mask_code,
    parse_note,
    split_frontmatter,
)

FIX = Path(__file__).parent / "fixtures" / "vault"


def _note(rel: str):
    return parse_note("fixture", rel, (FIX / rel).read_text(encoding="utf-8"), mtime=1_760_000_000)


def test_frontmatter_fields_and_core_keys():
    n = _note("10-Projects/dayo/dayo-notes-server.md")
    assert n.note_id == "202610021545"
    assert n.doc_id == "fixture:202610021545"
    assert (n.type, n.status) == ("folder-note", "active")
    assert n.created == "2026-10-02" and n.modified == "2026-10-20"
    assert n.aliases == ["dayo", "DayO"]
    assert n.folder == "10-Projects/dayo"
    assert n.meta == {"deadline": "2026-12-01"}  # custom keys land in the flattened field


def test_tags_normalised_from_list_string_and_inline():
    assert _note("10-Projects/dayo/dayo-notes-server.md").tags == ["projects/dayo", "mcp"]
    n = _note("30-Resources/search/Elasticsearch analyzers.md")
    assert n.tags == ["ai/rag", "search", "search/tuning"]  # string tags split; inline ignores code


def test_links_internal_external_and_code_ignored():
    n = _note("10-Projects/dayo/dayo-notes-server.md")
    assert n.links_out == [
        "elasticsearch analyzers",
        "30-resources/search/elasticsearch analyzers",
        "vault-discovery",
    ]
    assert n.urls == ["https://example.com/docs"]
    m = _note("30-Resources/search/Elasticsearch analyzers.md")
    assert m.links_out == ["../dayo/dayo-notes-server"]  # png embed, code-block links dropped


def test_note_without_frontmatter_uses_path_id_and_mtime():
    n = _note("00-Inbox/ideia-solta.md")
    assert n.doc_id == "fixture:path:00-Inbox/ideia-solta.md"
    assert n.title == "ideia-solta" and n.type is None
    assert n.modified == "2025-10-09"
    assert n.tags == ["chatbot"]


def test_title_precedence_h1_over_filename():
    note = _note("30-Resources/search/Elasticsearch analyzers.md")
    assert note.title == "Elasticsearch analyzers for PT/EN notes"


def test_invalid_frontmatter_is_survivable():
    meta, body, warns = split_frontmatter("---\nkey: [unclosed\n---\nhello\n")
    assert meta == {} and body.strip() == "hello" and warns
    n = parse_note("v", "a.md", "---\nkey: [unclosed\n---\n# T\ntext")
    assert n.title == "T" and n.warnings


def test_chunks_follow_headings_and_skip_code_fences():
    chunks = chunk_markdown("intro\n\n# A\nalpha\n\n```\n# not a heading\n```\n\n## B\nbeta\n")
    assert [(c.heading, c.level) for c in chunks] == [("", 0), ("A", 1), ("B", 2)]
    assert "# not a heading" in chunks[1].text
    assert [c.position for c in chunks] == [0, 1, 2]


def test_long_section_is_split_under_max_chars():
    para = "word " * 120
    chunks = chunk_markdown("# H\n" + "\n\n".join([para] * 5), max_chars=700)
    assert len(chunks) > 1 and all(len(c.text) <= 700 for c in chunks)


def test_mask_code_and_extract_links():
    raw = "a `[[x]]` b\n```\n[[y]]\n```\n[[z|alias]] [t](p%20q.md#h) [m](mailto:a@b.c)"
    masked = mask_code(raw)
    assert extract_links(masked) == (["z", "p q"], [])


def test_content_hash_changes_with_content():
    a = parse_note("v", "a.md", "one")
    b = parse_note("v", "a.md", "two")
    assert a.content_hash != b.content_hash
    assert a.content_hash == parse_note("v", "a.md", "one").content_hash


def test_content_hash_changes_with_pipeline_versions_and_params(monkeypatch):
    raw = "# Heading\n\n" + ("paragraph text here. " * 40) + "\n\n" + ("second paragraph. " * 40)
    base = parse_note("v", "a.md", raw, chunk_max_chars=1500)

    # Chunk parameter changes automatically invalidate content_hash
    smaller = parse_note("v", "a.md", raw, chunk_max_chars=400)
    with_overlap = parse_note("v", "a.md", raw, chunk_max_chars=400, chunk_overlap=80)
    with_emb = parse_note(
        "v", "a.md", raw, chunk_max_chars=1500, embedding_model="nomic-embed-text"
    )
    hashes = {
        base.content_hash,
        smaller.content_hash,
        with_overlap.content_hash,
        with_emb.content_hash,
    }
    assert len(hashes) == 4

    # Bumping PARSER_VERSION or CHUNKER_VERSION invalidates content_hash even if chunks match
    monkeypatch.setattr(parser_mod, "PARSER_VERSION", 2)
    bumped_parser = parse_note("v", "a.md", raw, chunk_max_chars=1500)
    assert bumped_parser.content_hash != base.content_hash

    monkeypatch.setattr(parser_mod, "PARSER_VERSION", 1)
    monkeypatch.setattr(parser_mod, "CHUNKER_VERSION", 2)
    bumped_chunker = parse_note("v", "a.md", raw, chunk_max_chars=1500)
    assert bumped_chunker.content_hash != base.content_hash


def test_content_hash_detects_chunker_output_change_without_version_bump(monkeypatch):
    # Even if CHUNKER_VERSION is not bumped, a change in produced chunks changes content_hash
    def custom_chunker(body: str, max_chars: int = 1500, overlap: int = 0) -> list[Chunk]:
        return [Chunk(position=0, heading="Custom", level=1, text=body.strip())]

    monkeypatch.setitem(CHUNKERS, "heading", custom_chunker)
    modified = parse_note("v", "a.md", "# Real Heading\n\nbody")
    monkeypatch.undo()
    original = parse_note("v", "a.md", "# Real Heading\n\nbody")
    assert modified.content_hash != original.content_hash


def test_chunk_overlap_and_registry_validation():
    para1 = "alpha " * 60
    para2 = "beta " * 60
    chunks_no_overlap = chunk_markdown(f"# H\n{para1}\n\n{para2}", max_chars=450, overlap=0)
    chunks_overlap = chunk_markdown(f"# H\n{para1}\n\n{para2}", max_chars=450, overlap=60)
    assert len(chunks_overlap) >= 2
    assert "alpha" in chunks_overlap[1].text and "alpha" not in chunks_no_overlap[1].text

    with pytest.raises(ValueError, match="unknown chunker"):
        get_chunker("nope")
    with pytest.raises(ValueError, match="unknown parser"):
        get_parser("nope")
