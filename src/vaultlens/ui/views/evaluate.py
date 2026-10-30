"""Evaluate page (wireframe: Evaluate). Judged queries -> _rank_eval -> nDCG@10 per strategy."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from vaultlens.evaluate import load_judgments, run_evaluation
from vaultlens.search.query_builder import STRATEGIES
from vaultlens.ui import services

be = services.backend()
st.title("Evaluate")
path = st.text_input("Judgments file", "config/judged_queries.yaml")

if not Path(path).exists():
    st.warning(f"{path} not found.")
    st.stop()
try:
    judged = load_judgments(path)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Judged set: {len(judged)} queries · metric: nDCG@10")
if not judged:
    st.info("No judged queries yet. Add some to the file (format: docs/SPEC.md > Evaluate).")
    st.stop()

chosen = st.multiselect("Strategies", list(STRATEGIES), default=list(STRATEGIES), format_func=lambda k: f"{k}: {STRATEGIES[k]['label']}")
if st.button("Run evaluation", type="primary") and chosen:
    try:
        results = run_evaluation(be, judged, chosen)
    except Exception as exc:
        st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
        st.stop()
    rows = [{"query": q.query, **{f"{r.strategy}: {r.label}": round(r.per_query.get(q.id, 0.0), 3) for r in results}} for q in judged]
    rows.append({"query": "Mean", **{f"{r.strategy}: {r.label}": round(r.mean, 3) for r in results}})
    st.dataframe(rows, hide_index=True, use_container_width=True)
