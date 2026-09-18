"""
HTML loader — parses HTML files with BeautifulSoup, removes metadata and
non-content markup, cleans the extracted text, and returns clean chunks.

Pipeline:
    1. Parse HTML (lxml if available, html.parser fallback).
    2. Strip metadata: <head>, scripts, styles, comments, nav/footer/aside,
       forms, iframes, hidden elements, etc.
    3. Extract readable text while preserving block structure (headings,
       paragraphs, lists, tables, <pre> blocks).
    4. Clean text: normalize whitespace, drop empty lines, remove control chars.
    5. Chunk: split on natural boundaries (paragraphs -> sentences) with overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import (
    BeautifulSoup,
    CData,
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    ProcessingInstruction,
    Tag,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

# Parser preference: lxml is faster and more forgiving; html.parser always works.
try:
    import lxml  # noqa: F401

    DEFAULT_PARSER = "lxml"
except ImportError:
    DEFAULT_PARSER = "html.parser"


# --- Tags that carry no usable document content (metadata / boilerplate) ----
STRIP_TAGS = [
    "head",
    "script",
    "style",
    "noscript",
    "template",
    "meta",
    "link",
    "title",
    "base",
    "iframe",
    "svg",
    "canvas",
    "form",
    "input",
    "button",
    "select",
    "option",
    "textarea",
    "nav",
    "aside",
    "footer",
    "header",
    "menu",
    "object",
    "embed",
    "audio",
    "video",
]

# Block-level elements: force a line break after them so paragraphs survive.
BLOCK_TAGS = [
    "address",
    "article",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
]

# Hidden content via CSS is noise for a text pipeline.
HIDDEN_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
MULTI_BLANK_RE = re.compile(r"\n{3,}")
SPACES_RE = re.compile(r"[ \t\f\v]+")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class HTMLChunk:
    """A single clean chunk of extracted HTML text."""

    text: str
    index: int
    source: str | None = None
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def _soup_from(raw: str | bytes) -> BeautifulSoup:
    """Parse raw HTML, preferring lxml and falling back to html.parser."""
    try:
        return BeautifulSoup(raw, "lxml")
    except Exception:
        return BeautifulSoup(raw, "html.parser")


def strip_metadata(soup: BeautifulSoup) -> None:
    """Remove metadata tags, comments, doctype, hidden/boilerplate elements."""
    # Comments, CDATA, processing instructions, doctype declarations.
    noise_strings = (Comment, CData, ProcessingInstruction, Declaration, Doctype)
    for node in soup.find_all(string=lambda s: isinstance(s, noise_strings)):
        node.extract()

    for name in STRIP_TAGS:
        for tag in soup.find_all(name):
            tag.decompose()

    # Elements hidden via inline CSS.
    for tag in soup.find_all(style=HIDDEN_RE):
        tag.decompose()

    # Common "skip to content"/screen-reader-only markers.
    for cls in ("sr-only", "screen-reader-only", "visually-hidden", "skip-link"):
        for tag in soup.find_all(class_=cls):
            tag.decompose()


def _table_to_text(table: Tag) -> str:
    """Render a table as pipe-separated rows so structure survives chunking."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_text(soup: BeautifulSoup) -> str:
    """Extract readable text, preserving block structure and tables."""
    # Tables are replaced with their textual representation before generic
    # extraction so cell boundaries are not glued together.
    for table in soup.find_all("table"):
        table.replace_with(NavigableString("\n" + _table_to_text(table) + "\n"))

    # <br> becomes a newline; block elements are separated by newlines.
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))

    body = soup.body or soup
    lines: list[str] = []

    def walk(node: Tag | NavigableString) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child)
                if text.strip():
                    lines.append(" ".join(text.split()))
            elif isinstance(child, Tag):
                is_pre = child.name == "pre"
                if is_pre:
                    lines.append(child.get_text())  # keep raw layout
                else:
                    walk(child)
                if child.name in BLOCK_TAGS:
                    lines.append("\n")

    walk(body)
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters / empty lines."""
    text = CONTROL_CHARS_RE.sub("", text)
    # Normalize unicode spaces to ASCII, unify quotes/dashes.
    text = (
        text.replace("\u00a0", " ")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    # Collapse intra-line whitespace but keep newlines.
    text = "\n".join(SPACES_RE.sub(" ", line).strip() for line in text.splitlines())
    # Trim very short junk lines (stray bullets/punctuation) and blanks.
    lines = [line for line in text.splitlines() if len(line) > 1]
    text = "\n".join(lines)
    return MULTI_BLANK_RE.sub("\n\n", text).strip()


def split_long_block(block: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph by sentences, then hard-wrap if needed."""
    if len(block) <= max_chars:
        return [block]

    parts: list[str] = []
    current = ""
    for sentence in SENTENCE_END_RE.split(block):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(sentence) > max_chars:  # pathological: no sentence ends
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:].lstrip()
        current = sentence
    if current:
        parts.append(current)
    return parts


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 40,
) -> list[str]:
    """
    Chunk cleaned text on natural boundaries.

    Paragraphs are packed up to ``chunk_size``; oversized paragraphs fall back
    to sentence splits; a tail of the previous chunk is carried over as
    ``chunk_overlap`` context.
    """
    if not text:
        return []

    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        stripped = current.strip()
        if len(stripped) >= min_chunk_chars:
            chunks.append(stripped)
        current = ""

    for block in blocks:
        for piece in split_long_block(block, chunk_size):
            if not current:
                current = piece
            elif len(current) + len(piece) + 2 <= chunk_size:
                current = f"{current}\n{piece}"
            else:
                flush()
                if chunk_overlap > 0 and chunks:
                    tail = chunks[-1][-chunk_overlap:]
                    cut = tail.find(" ")
                    tail = tail[cut + 1 :] if cut != -1 else tail
                    current = f"{tail}\n{piece}".strip()
                else:
                    current = piece
    flush()
    return chunks


class HTMLLoader:
    """Load HTML files, clean them, and return clean text chunks."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        min_chunk_chars: int = 40,
        parser: str | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.parser = parser or DEFAULT_PARSER

    # -- public API ---------------------------------------------------------

    def load_text(self, raw: str | bytes, source: str | None = None) -> list[str]:
        """Parse raw HTML and return a list of clean text chunks."""
        soup = _soup_from(raw)
        strip_metadata(soup)
        text = clean_text(extract_text(soup))
        if not text:
            logger.warning(f"No readable text extracted from {source or 'raw HTML'}")
            return []
        return chunk_text(text, self.chunk_size, self.chunk_overlap, self.min_chunk_chars)

    def load(self, path: str | Path) -> list[HTMLChunk]:
        """Load an HTML file and return chunk objects with provenance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")
        raw = path.read_bytes()
        texts = self.load_text(raw, source=str(path))
        logger.info(f"{path.name}: {len(texts)} chunks")
        return [HTMLChunk(text=t, index=i, source=str(path)) for i, t in enumerate(texts)]


def parse_html_file(path: str | Path, **kwargs) -> list[str]:
    """Convenience one-shot: parse an HTML file and return clean text chunks."""
    return [chunk.text for chunk in HTMLLoader(**kwargs).load(path)]
