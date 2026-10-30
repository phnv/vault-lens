import re

from vaultlens.models import Note
from vaultlens.search.mapping import index_name, load_mapping, mapped_fields, version_of
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
    note = Note(vault="v", path="a.md", doc_id="v:1", title="t", body="b", content_hash="h", folder="")
    top_level = {k for k in mapped_fields(load_mapping()) if "." not in k}
    assert set(note.to_es_doc()) == top_level


def test_index_naming():
    assert index_name("notes", 3) == "notes_v3" and version_of("notes_v3") == 3 and version_of("notes") == 0
