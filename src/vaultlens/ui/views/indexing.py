"""Vaults and Indexing page (wireframe: Vaults and Indexing)."""
from __future__ import annotations

import streamlit as st

from vaultlens.indexing.service import IndexCompatibilityError
from vaultlens.search.mapping import MAPPING_VERSION, load_mapping, mapped_fields
from vaultlens.ui import services
from vaultlens.vault.parser import CHUNKER_VERSION, PARSER_VERSION

cfg = services.settings()
be = services.backend()
store = services.store()
svc = services.index_service()
st.title("Vaults and Indexing")

try:
    stats = be.stats()
    compat_ok, compat_msg = be.check_index_compatibility(cfg.embedding_model)
except Exception as exc:
    st.error(f"Elasticsearch unreachable ({type(exc).__name__}). Run `make up`.")
    st.stop()

if not compat_ok:
    st.error(compat_msg)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", f"{stats['docs']:,}")
m2.metric("Chunks", f"{stats['chunks']:,}")
m3.metric("Index size", f"{stats['size_bytes'] / 1_048_576:.1f} MB")
m4.metric("Alias", f"{cfg.es_alias} → {stats['index'] or 'none'}")

st.caption(
    f"Pipeline: parser=`{cfg.parser}` (v{PARSER_VERSION}) · "
    f"chunker=`{cfg.chunker}` (v{CHUNKER_VERSION}, "
    f"max={cfg.chunk_max_chars}, overlap={cfg.chunk_overlap}) · "
    f"embedding=`{cfg.embedding_model}` · mapping=v{MAPPING_VERSION}"
)

st.subheader("Registered vaults")
force_sync = st.checkbox(
    "Force sync (re-index unchanged notes in-place)",
    value=False,
    help="Re-index all notes in the selected vault even when parsed output and hashes match.",
)
state = store.sync_state()
for v in cfg.vaults:
    rep = state.get(v.slug, {})
    c1, c2, c3, c4 = st.columns([2, 3, 3, 1])
    c1.markdown(f"**{v.slug}**")
    c2.caption(v.path)
    c3.caption(
        f"{rep.get('scanned', '–')} notes · last sync {rep.get('finished_at', 'never')} · "
        f"+{rep.get('new', 0)} new, {rep.get('changed', 0)} changed, "
        f"{rep.get('removed', 0)} removed"
    )
    if c4.button("Sync", key=f"sync_{v.slug}"):
        try:
            with st.spinner(f"Syncing {v.slug}…"):
                r = svc.sync(v.slug, force=force_sync)
            st.success(
                f"{v.slug}: +{r.new} new · {r.changed} changed · "
                f"{r.removed} removed in {r.seconds}s"
            )
            if r.errors:
                st.warning(f"{len(r.errors)} errors, first: {r.errors[0]}")
        except IndexCompatibilityError as exc:
            st.error(str(exc))
st.caption(
    "Adding a vault: mount it read-only in docker-compose.yml, "
    "then add it to config/vaultlens.yaml."
)

st.subheader("Reindex")
st.caption("Full reindex builds the next notes_vN from all vaults, then swaps the alias.")
if st.button("Full reindex"):
    with st.spinner("Reindexing…"):
        reports = svc.full_reindex()
    st.success("Done: " + ", ".join(f"{r.vault} {r.scanned} notes" for r in reports))

st.subheader("Mapping inspector")
st.dataframe(
    [{"field": f, "type": t} for f, t in mapped_fields(load_mapping()).items()],
    hide_index=True, use_container_width=True,
)
