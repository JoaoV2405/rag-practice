"""Low-level helpers for the RAG pipeline: PDF discovery, text extraction,
and chunking. Kept out of the notebook so the notebook cells can focus on
the actual RAG steps (embedding, indexing, retrieval, generation).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


def iter_pdfs(root: Path) -> list[Path]:
    """Recursively find every PDF under `root`, sorted for reproducibility."""
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def extract_pages(pdf_path: Path) -> list[str]:
    """Return a list of page texts (1 entry per page) for a PDF."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text") for page in doc]


def _split_paragraphs(text: str) -> list[str]:
    # Collapse hyphenation/line-wrap artifacts a bit, then split on blank lines.
    text = re.sub(r"[ \t]+\n", "\n", text)
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Paragraph-aware sliding-window chunker.

    Greedily packs paragraphs into windows of ~chunk_size characters. A
    paragraph longer than chunk_size is hard-split. Consecutive chunks share
    `overlap` characters of trailing context so retrieval doesn't lose
    information that sits right at a chunk boundary.
    """
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Hard-split an overlong paragraph into chunk_size windows.
            for i in range(0, len(para), chunk_size - overlap):
                piece = para[i : i + chunk_size]
                if current and len(current) + 1 + len(piece) <= chunk_size:
                    current = f"{current}\n{piece}"
                else:
                    flush()
                    current = piece
            continue

        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            flush()
            # Seed the new chunk with overlap from the end of the previous one.
            tail = chunks[-1][-overlap:] if chunks and overlap else ""
            current = f"{tail}\n{para}".strip() if tail else para

    flush()
    return chunks


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer used for BM25 (language-agnostic, no stemming)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ChunkRecord:
    id: str
    text: str
    source_path: str
    category: str
    paper: str
    page: int
    chunk_index: int


def build_chunk_records(
    pdf_path: Path,
    data_root: Path,
    pages: list[str],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[ChunkRecord]:
    """Chunk every page of a PDF and attach citation metadata to each chunk."""
    rel_path = pdf_path.relative_to(data_root)
    category = str(rel_path.parent) if rel_path.parent != Path(".") else "root"
    paper = pdf_path.stem

    records: list[ChunkRecord] = []
    for page_num, page_text in enumerate(pages, start=1):
        for chunk_idx, chunk in enumerate(chunk_text(page_text, chunk_size, overlap)):
            raw_id = f"{rel_path}::p{page_num}::c{chunk_idx}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
            records.append(
                ChunkRecord(
                    id=chunk_id,
                    text=chunk,
                    source_path=str(rel_path),
                    category=category,
                    paper=paper,
                    page=page_num,
                    chunk_index=chunk_idx,
                )
            )
    return records
