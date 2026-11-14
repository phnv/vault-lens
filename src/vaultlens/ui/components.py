"""Shared Streamlit widgets: custom-field (meta.*) and date filter widgets, Clear helpers."""

from __future__ import annotations

from datetime import date

import streamlit as st

from vaultlens.search.query_builder import (
    DATE_FIELDS,
    date_range_spec,
    meta_filters,
    meta_rows,
)
from vaultlens.ui import services

ss = st.session_state


def _rows_key(prefix: str) -> str:
    return f"{prefix}_meta_rows"


def set_meta_rows(prefix: str, filters: dict[str, list[str]]) -> None:
    """Load meta.* entries of a filters dict into the editor state (safe inside callbacks)."""
    rows = meta_rows(filters)
    ss[_rows_key(prefix)] = list(range(len(rows)))
    ss[f"{prefix}_meta_next"] = len(rows)
    for i, (key, values) in enumerate(rows):
        ss[f"{prefix}_mk_{i}"] = key
        ss[f"{prefix}_mv_{i}"] = values


def _add_row(prefix: str) -> None:
    n = ss.get(f"{prefix}_meta_next", 0)
    ss[_rows_key(prefix)] = [*ss.get(_rows_key(prefix), []), n]
    ss[f"{prefix}_meta_next"] = n + 1


def _remove_row(prefix: str, i: int) -> None:
    ss[_rows_key(prefix)] = [r for r in ss.get(_rows_key(prefix), []) if r != i]
    ss.pop(f"{prefix}_mk_{i}", None)
    ss.pop(f"{prefix}_mv_{i}", None)


def meta_filters_from_state(prefix: str) -> dict[str, list[str]]:
    """SearchSpec.filters entries from the editor's session state (usable before it renders)."""
    rows = [
        (ss.get(f"{prefix}_mk_{i}", "") or "", ss.get(f"{prefix}_mv_{i}", []) or [])
        for i in ss.get(_rows_key(prefix), [])
    ]
    return meta_filters(rows)


def meta_filter_editor(prefix: str, vault: str | None) -> None:
    """Rows of (key, value dropdown); AND across rows. Read via meta_filters_from_state."""
    st.markdown("**Custom fields**")
    for i in ss.get(_rows_key(prefix), []):
        c1, c2 = st.columns([5, 1])
        key = c1.text_input(
            "Key",
            key=f"{prefix}_mk_{i}",
            placeholder="key, e.g. course",
            label_visibility="collapsed",
        )
        c2.button(
            "✕", key=f"{prefix}_mx_{i}", on_click=_remove_row, args=(prefix, i), help="Remove"
        )
        counts: dict[str, int] = {}
        if key.strip():
            try:
                counts = dict(services.meta_values(key.strip(), vault))
            except Exception:
                st.caption(f"Values for '{key.strip()}' unavailable.")
        selected = ss.get(f"{prefix}_mv_{i}", [])
        options = list(dict.fromkeys([*counts, *selected]))
        st.multiselect(
            "Values",
            options,
            key=f"{prefix}_mv_{i}",
            label_visibility="collapsed",
            placeholder="values" if options else "no values for this key",
            format_func=lambda v, c=counts: f"{v} ({c[v]})" if v in c else v,
        )
    st.button("＋ Add field", key=f"{prefix}_meta_add", on_click=_add_row, args=(prefix,))


def clear_meta(prefix: str) -> None:
    """Callback helper: drop all meta rows for a page prefix."""
    for i in ss.get(_rows_key(prefix), []):
        ss.pop(f"{prefix}_mk_{i}", None)
        ss.pop(f"{prefix}_mv_{i}", None)
    ss[_rows_key(prefix)] = []


# date filter ----------------------------------------------------------------------------------


def date_filter_widgets(prefix: str) -> None:
    """One field dropdown (created/modified) plus a date-range picker."""
    st.markdown("**Date**")
    st.selectbox(
        "Date field",
        sorted(DATE_FIELDS),
        index=sorted(DATE_FIELDS).index("modified"),
        key=f"{prefix}_date_field",
        label_visibility="collapsed",
    )
    st.date_input(
        "Date range",
        value=(),
        key=f"{prefix}_dates",
        format="YYYY-MM-DD",
        label_visibility="collapsed",
    )


def date_range_from_state(prefix: str) -> dict[str, str] | None:
    dates = ss.get(f"{prefix}_dates", ())
    dates = (dates,) if isinstance(dates, date) else tuple(dates or ())
    return date_range_spec(ss.get(f"{prefix}_date_field", "modified"), dates)


def set_date_range(prefix: str, date_range: dict[str, str] | None) -> None:
    """Restore a saved date_range into the widgets (safe inside callbacks)."""
    dr = date_range or {}
    ss[f"{prefix}_date_field"] = dr.get("field", "modified")
    try:
        bounds = tuple(date.fromisoformat(dr[k][:10]) for k in ("gte", "lte") if dr.get(k))
    except ValueError:
        bounds = ()
    ss[f"{prefix}_dates"] = bounds


def clear_date(prefix: str) -> None:
    set_date_range(prefix, None)
