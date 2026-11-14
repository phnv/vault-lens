"""Markdown note -> Note. Pure functions, stdlib + PyYAML only. Never touches the filesystem."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime
from posixpath import dirname
from typing import Any
from urllib.parse import unquote

import yaml

from ..models import Chunk, Note

PARSER_VERSION = 1
CHUNKER_VERSION = 1
PARSER_VERSION = 1
CHUNKER_VERSION = 1
CORE_KEYS = {
    "id",
    "title",
    "aliases",
    "alias",
    "tags",
    "tag",
    "type",
    "status",
    "created",
    "modified",
}
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_WIKILINK = re.compile(r"!?\[\[([^\]\n]+?)\]\]")
_MDLINK = re.compile(r'(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+"[^"\n]*")?\)')
_INLINE_TAG = re.compile(r"(?<![\w/&#(])#([A-Za-z\u00C0-\u024F_][\w/\-\u00C0-\u024F]*)")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_ATTACHMENT_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".mp3",
    ".mp4",
    ".webm",
    ".mov",
    ".wav",
    ".zip",
    ".csv",
    ".xlsx",
    ".docx",
    ".pptx",
}


def split_frontmatter(raw: str) -> tuple[dict[str, Any], str, list[str]]:
    text = raw.lstrip("\ufeff")
    if not (text.startswith("---\n") or text.startswith("---\r\n")):
        return {}, text, []
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            body = "".join(lines[i + 1 :])
            try:
                data = yaml.safe_load("".join(lines[1:i])) or {}
            except yaml.YAMLError as exc:
                return {}, body, [f"frontmatter: invalid YAML ({type(exc).__name__})"]
            if not isinstance(data, dict):
                return {}, body, ["frontmatter: not a mapping"]
            return data, body, []
    return {}, text, []


def mask_code(text: str) -> str:
    """Blank out fenced and inline code so links/tags inside code are ignored."""
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            out.append("")
        elif m:
            fence = m.group(1)
            out.append("")
        else:
            out.append(re.sub(r"`[^`\n]*`", "", line))
    return "\n".join(out)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[,\n]", value)
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value if v is not None]
    else:
        items = [str(value)]
    return [i.strip() for i in items if i and i.strip()]


def _norm_tags(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        for piece in v.split():  # "a b" in a string tag field means two tags
            tag = piece.lstrip("#").strip().lower()
            if tag:
                seen.setdefault(tag, None)
    return list(seen)


def _date_str(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and _DATE.match(value.strip()):
        return value.strip()[:10]
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value if v is not None]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _ext(target: str) -> str:
    last = target.rsplit("/", 1)[-1]
    return last[last.rfind(".") :].lower() if "." in last else ""


def extract_links(masked_body: str) -> tuple[list[str], list[str]]:
    """Return (internal link targets lowercased, external urls). Attachments are dropped."""
    internal: dict[str, None] = {}
    urls: dict[str, None] = {}
    for m in _WIKILINK.finditer(masked_body):
        target = m.group(1).split("|")[0].split("#")[0].split("^")[0].strip()
        if not target:
            continue
        ext = _ext(target)
        if ext in _ATTACHMENT_EXT:
            continue
        internal.setdefault((target[:-3] if ext == ".md" else target).lower(), None)
    for m in _MDLINK.finditer(masked_body):
        target = m.group(1).strip("<>")
        low = target.lower()
        if low.startswith(("http://", "https://")):
            urls.setdefault(target, None)
            continue
        if low.startswith(("mailto:", "#", "tel:")) or "://" in low:
            continue
        path = unquote(target.split("#")[0])
        ext = _ext(path)
        if not path or (ext and ext != ".md"):
            continue
        internal.setdefault((path[:-3] if ext == ".md" else path).lower(), None)
    return list(internal), list(urls)


def _overlap_prefix(prev: str, overlap: int, max_chars: int, next_len: int) -> str:
    if overlap <= 0 or not prev:
        return ""
    budget = min(overlap, max(0, max_chars - next_len - 2))
    if budget <= 0:
        return ""
    tail = prev[-budget:]
    if len(prev) > budget:
        space = tail.find(" ")
        if space != -1 and space < len(tail) - 1:
            tail = tail[space + 1 :]
    return tail.strip()


def _split_long(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        while len(para) > max_chars:  # hard split on whitespace
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if buf:
                parts.append(buf)
                buf = ""
            parts.append(para[:cut].strip())
            step = max(1, cut - max(0, overlap)) if overlap > 0 else cut
            para = para[step:].strip()
        if buf and len(buf) + len(para) + 2 > max_chars:
            parts.append(buf)
            prefix = _overlap_prefix(buf, overlap, max_chars, len(para))
            buf = f"{prefix}\n\n{para}" if prefix else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        parts.append(buf)
    return [p for p in parts if p.strip()]


def chunk_markdown(body: str, max_chars: int = 1500, overlap: int = 0) -> list[Chunk]:
    """Split on ATX headings (outside code fences); long sections split on paragraphs."""
    sections: list[tuple[int, str, list[str]]] = []
    level, heading, lines = 0, "", []
    fence: str | None = None
    for line in body.splitlines():
        m = _FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if m:
            fence = m.group(1)
            lines.append(line)
            continue
        h = _HEADING.match(line)
        if h:
            sections.append((level, heading, lines))
            level, heading, lines = len(h.group(1)), h.group(2).strip(), []
        else:
            lines.append(line)
    sections.append((level, heading, lines))

    chunks: list[Chunk] = []
    for lvl, head, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if not text:
            continue
        for piece in _split_long(text, max_chars, overlap):
            chunks.append(Chunk(position=len(chunks), heading=head, level=lvl, text=piece))
    return chunks


CHUNKERS: dict[str, Callable[..., list[Chunk]]] = {
    "heading": chunk_markdown,
}


def get_chunker(name: str) -> Callable[..., list[Chunk]]:
    if name not in CHUNKERS:
        raise ValueError(f"unknown chunker: {name!r} (available: {sorted(CHUNKERS)})")
    return CHUNKERS[name]


def make_doc_id(vault: str, note_id: str | None, path: str) -> str:
    return f"{vault}:{note_id}" if note_id else f"{vault}:path:{path}"


def _first_h1(masked: str) -> str | None:
    for line in masked.splitlines():
        m = _HEADING.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2)
    return None


def compute_content_hash(
    *,
    doc_id: str,
    title: str,
    body: str,
    folder: str,
    note_id: str | None,
    aliases: list[str],
    tags: list[str],
    note_type: str | None,
    status: str | None,
    created: str | None,
    modified: str | None,
    links_out: list[str],
    urls: list[str],
    meta: dict[str, Any],
    chunks: list[Chunk],
    size_bytes: int,
    parser: str = "default",
    parser_version: int = PARSER_VERSION,
    chunker: str = "heading",
    chunker_version: int = CHUNKER_VERSION,
    chunk_max_chars: int = 1500,
    chunk_overlap: int = 0,
    embedding_model: str = "off",
) -> str:
    """Canonical SHA-256 digest of parsed Note output + chunks + active pipeline versions/config."""
    payload = {
        "pipeline": {
            "parser": parser,
            "parser_version": parser_version,
            "chunker": chunker,
            "chunker_version": chunker_version,
            "chunk_max_chars": chunk_max_chars,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
        },
        "note": {
            "doc_id": doc_id,
            "title": title,
            "body": body,
            "folder": folder,
            "note_id": note_id,
            "aliases": aliases,
            "tags": tags,
            "type": note_type,
            "status": status,
            "created": created,
            "modified": modified,
            "links_out": links_out,
            "urls": urls,
            "meta": meta,
            "size_bytes": size_bytes,
            "chunks": [asdict(c) for c in chunks],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8", "replace"
    )
    return hashlib.sha256(encoded).hexdigest()


def parse_note(
    vault: str,
    rel_path: str,
    raw: str,
    mtime: float | None = None,
    chunk_max_chars: int = 1500,
    *,
    chunker: str = "heading",
    chunk_overlap: int = 0,
    parser: str = "default",
    embedding_model: str = "off",
) -> Note:
    fm, body, warnings = split_frontmatter(raw)
    masked = mask_code(body)

    stem = rel_path.rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.lower().endswith(".md") else stem
    title = str(fm.get("title") or _first_h1(masked) or stem).strip()

    raw_id = fm.get("id")
    note_id = str(raw_id).strip() if raw_id not in (None, "") else None

    fm_tags = _as_list(fm.get("tags", fm.get("tag")))
    inline_tags = [m.group(1) for m in _INLINE_TAG.finditer(masked)]
    internal, urls = extract_links(masked)

    modified = _date_str(fm.get("modified"))
    if modified is None and mtime is not None:
        modified = datetime.fromtimestamp(mtime, UTC).date().isoformat()

    extra = {k: _json_safe(v) for k, v in fm.items() if k not in CORE_KEYS and v is not None}
    doc_id = make_doc_id(vault, note_id, rel_path)
    clean_body = body.strip()
    folder = dirname(rel_path)
    aliases = _as_list(fm.get("aliases", fm.get("alias")))
    tags = _norm_tags(fm_tags + inline_tags)
    note_type = str(fm["type"]).strip() if fm.get("type") else None
    status = str(fm["status"]).strip() if fm.get("status") else None
    created = _date_str(fm.get("created"))
    size_bytes = len(raw.encode("utf-8", "replace"))
    chunker_fn = get_chunker(chunker)
    chunks = chunker_fn(body, chunk_max_chars, chunk_overlap)

    content_hash = compute_content_hash(
        doc_id=doc_id,
        title=title,
        body=clean_body,
        folder=folder,
        note_id=note_id,
        aliases=aliases,
        tags=tags,
        note_type=note_type,
        status=status,
        created=created,
        modified=modified,
        links_out=internal,
        urls=urls,
        meta=extra,
        chunks=chunks,
        size_bytes=size_bytes,
        parser=parser,
        parser_version=PARSER_VERSION,
        chunker=chunker,
        chunker_version=CHUNKER_VERSION,
        chunk_max_chars=chunk_max_chars,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )

    return Note(
        vault=vault,
        path=rel_path,
        doc_id=doc_id,
        title=title,
        body=clean_body,
        content_hash=content_hash,
        folder=folder,
        note_id=note_id,
        aliases=aliases,
        tags=tags,
        type=note_type,
        status=status,
        created=created,
        modified=modified,
        links_out=internal,
        urls=urls,
        meta=extra,
        chunks=chunks,
        size_bytes=size_bytes,
        warnings=warnings,
    )


PARSERS: dict[str, Callable[..., Note]] = {
    "default": parse_note,
}


def get_parser(name: str) -> Callable[..., Note]:
    if name not in PARSERS:
        raise ValueError(f"unknown parser: {name!r} (available: {sorted(PARSERS)})")
    return PARSERS[name]
