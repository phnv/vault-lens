import re

from vaultlens.models import Note
from vaultlens.search.mapping import (
    MAPPING_VERSION,
    check_meta_compatibility,
    index_name,
    load_mapping,
    mapped_fields,
    resolve_index_meta,
    version_of,
)
from vaultlens.search.query_builder import ALLOWED_FILTER_FIELDS, ALLOWED_QUERY_FIELDS


def test_every_referenced_analyzer_is_defined():
    m = load_mapping()
    defined = set(m["settings"]["analysis"]["analyzer"])
    builtin = {"standard"}
    used = set(re.findall(r'"(?:search_)?analyzer":\s*"([^"]+)"', str(m).replace("'", '"')))
    assert used <= defined | builtin, used - defined - builtin


def test_query_and_filter_fields_exist_in_mapping():
    fields = mapped_fields(load_mapping())
    assert ALLOWED_QUERY_FIELDS <= set(fields)
    assert ALLOWED_FILTER_FIELDS <= set(fields)


def test_note_document_matches_strict_mapping():
    note = Note(
        vault="v", path="a.md", doc_id="v:1", title="t", body="b", content_hash="h", folder=""
    )
    top_level = {k for k in mapped_fields(load_mapping()) if "." not in k}
    assert set(note.to_es_doc()) == top_level


def test_index_naming():
    assert index_name("notes", 3) == "notes_v3"
    assert version_of("notes_v3") == 3 and version_of("notes") == 0


def test_mapping_meta_and_compatibility():
    m = load_mapping(embedding_model="bge-m3")
    assert m["mappings"]["_meta"] == {
        "mapping_version": MAPPING_VERSION,
        "embedding_model": "bge-m3",
    }

    # Missing index or matching _meta is compatible
    assert check_meta_compatibility(None, None)[0] is True
    meta_ok = {"mapping_version": 1, "embedding_model": "off"}
    assert check_meta_compatibility("notes_v1", meta_ok)[0] is True

    # Legacy index created before _meta defaults to version_of(index) and embedding_model="off"
    assert resolve_index_meta("notes_v1", None) == meta_ok
    assert (
        check_meta_compatibility("notes_v1", None, mapping_version=1, embedding_model="off")[0]
        is True
    )

    # Mismatched mapping_version or embedding_model fails fast with reindex instructions
    ok_ver, msg_ver = check_meta_compatibility(
        "notes_v1", {"mapping_version": 1}, mapping_version=2
    )
    assert not ok_ver and "mapping_version" in msg_ver and "reindex" in msg_ver.lower()

    ok_emb, msg_emb = check_meta_compatibility(
        "notes_v1", None, embedding_model="nomic-embed-text"
    )
    assert not ok_emb and "embedding_model" in msg_emb and "reindex" in msg_emb.lower()
