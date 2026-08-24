"""
The bridge: our cleaned blocks -> LangChain Documents.

This is the only custom code between SEC EDGAR and standard LangChain. Once a
filing is a list of `Document` objects, every downstream step -- chunking,
embedding, Chroma, BM25, EnsembleRetriever, reranking -- is a stock LangChain
component used the way the Week 2 session demoed it.

Why keep our own cleaner at all, when LangChain has HTML loaders
--------------------------------------------------------------
Because a 10-K is not an ordinary web page. Two problems a generic loader
does not solve:

1. Filings use <table> for real financial data AND for page layout. A generic
   loader flattens both into a text blob, so "Data Center 115,186 47,525"
   arrives with no indication of which number belongs to which year.

2. EDGAR puts the "$" in its own cell, so a row comes through as
   ['Data Center', '$', '115,186', '$', '47,525'] against a 4-column header.
   Left alone, the columns stop lining up entirely.

`clean.py` classifies tables, folds the currency cells back into their values,
and tags every block with the Item section it came from. That section tag is
what lets an answer cite "NVIDIA 2025-02-26, Item 7" instead of an opaque
chunk id -- and citations are the part of RAG that makes an answer checkable.

Everything after this file is off-the-shelf.
"""
from __future__ import annotations

from typing import List, Sequence

from .clean import Block


def blocks_to_documents(blocks: Sequence[Block]) -> List:
    """
    Convert cleaned blocks into LangChain Documents.

    The metadata travels with the chunk through splitting, embedding and
    retrieval, which is exactly what makes citation possible at the end --
    LangChain carries a Document's metadata onto every chunk split from it.

    `is_table` is carried as a plain bool because Chroma's metadata values
    must be str/int/float/bool -- a nested dict or list is rejected at
    insert time, which is an unhelpful error to hit at 2am.
    """
    from langchain_core.documents import Document

    docs = []
    for b in blocks:
        docs.append(Document(
            page_content=b.text,
            metadata={
                "company": b.company or "",
                "filing_date": b.filing_date or "",
                "section": b.section or "",
                "source_url": b.source_url or "",
                "is_table": bool(b.kind == "table"),
                "order": int(b.order),
            },
        ))
    return docs


def citation(doc) -> str:
    """A human-checkable source line: '[NVIDIA 2025-02-26 · Item 7 - MD&A]'."""
    m = getattr(doc, "metadata", {}) or {}
    bits = [str(m.get(k, "")).strip() for k in ("company", "filing_date", "section")]
    bits = [b for b in bits if b]
    return "[" + " · ".join(bits) + "]" if bits else "[source]"


def describe(docs: Sequence, n: int = 3) -> str:
    """
    Print a few chunks the way the retriever sees them.

    Worth running after every change to chunking. Numbers in a stats table can
    look healthy while the actual text is garbage; looking at three real chunks
    catches that in seconds.
    """
    lines = []
    for d in list(docs)[:n]:
        m = d.metadata
        kind = "TABLE" if m.get("is_table") else "text"
        lines.append(f"--- {m.get('company','?')} · {m.get('section','?')} · {kind} "
                     f"· {len(d.page_content)} chars")
        lines.append(d.page_content[:420].strip())
        lines.append("")
    return "\n".join(lines)
