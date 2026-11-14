"""Needs a live Elasticsearch (`make up`, or ES_URL).

First job of ROADMAP M1 is to run these and fix whatever the real server disagrees with.
"""

from pathlib import Path
from uuid import uuid4

import pytest

from vaultlens.config import Settings, VaultConfig
from vaultlens.models import SearchSpec
from vaultlens.search.results import parse_response

from .conftest import es_url

pytestmark = pytest.mark.integration
FIX = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    from vaultlens.indexing.service import IndexService
    from vaultlens.search.backend import SearchBackend
    from vaultlens.store import JsonStore

    alias = f"test_{uuid4().hex[:8]}"
    data = tmp_path_factory.mktemp("data")
    settings = Settings(
        es_url=es_url(),
        es_alias=alias,
        data_dir=str(data),
        vaults=(VaultConfig(slug="fixture", path=str(FIX)),),
    )
    backend = SearchBackend(es_url(), alias)
    yield backend, IndexService(backend, settings, JsonStore(str(data)))
    index = backend.current_index()
    if index:
        backend.es.indices.delete(index=index)


def _search(backend, **kw):
    return parse_response(backend.search(SearchSpec(**kw)))


def test_first_sync_indexes_everything_then_is_idempotent(stack):
    backend, svc = stack
    first = svc.sync("fixture")
    assert (first.scanned, first.new, first.changed, first.removed) == (
        3,
        3,
        0,
        0,
    ) and not first.errors
    second = svc.sync("fixture")
    assert (second.new, second.changed, second.removed, second.unchanged) == (0, 0, 0, 3)
    assert backend.stats()["docs"] == 3 and backend.stats()["chunks"] >= 3


def test_search_finds_note_with_highlight_and_best_chunks(stack):
    backend, _ = stack
    r = _search(backend, query="analyzer", vaults=["fixture"])
    assert r.total >= 1 and r.hits[0].path.endswith("Elasticsearch analyzers.md")
    assert r.hits[0].snippet and r.hits[0].chunks


def test_accent_and_case_insensitive_match_via_folding(stack):
    backend, _ = stack
    r = _search(backend, query="REUNIOES", fields={"body": 1.0}, vaults=["fixture"])
    assert any("analyzers" in h.path for h in r.hits)


def test_folder_tree_filter_matches_descendants_and_facets(stack):
    backend, _ = stack
    r = _search(backend, filters={"folder.tree": ["10-Projects"]})
    assert [h.path for h in r.hits] == ["10-Projects/dayo/dayo-notes-server.md"]
    assert ("10-Projects/dayo", 1) in _search(backend).facets["folder"]


def test_full_reindex_swaps_alias(stack):
    backend, svc = stack
    before = backend.current_index()
    svc.full_reindex()
    after = backend.current_index()
    st = backend.stats()
    assert after != before and st["docs"] == 3 and st["embedded_chunks"] == 0


def test_pt_folded_treats_accented_and_ascii_forms_identically(stack):
    backend, _ = stack
    for accented, ascii_form in (
        ("opções", "opcoes"),
        ("certificações", "certificacoes"),
        ("limitações", "limitacoes"),
    ):
        assert backend.analyze("pt_folded", accented) == backend.analyze("pt_folded", ascii_form)
