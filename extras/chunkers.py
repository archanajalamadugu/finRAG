"""
Two chunking strategies, built for a controlled comparison.

The Week 2 brief asks us to compare fixed-size vs semantic chunking on the
same queries. To make that comparison mean something, the two strategies here
differ in exactly one respect -- WHERE the boundaries are placed -- and share
everything else (same cleaned blocks, same metadata, same size ceiling, same
table policy). `atomic_tables` is exposed on both so table handling can be
ablated separately instead of silently confounding the result.

Strategy A  fixed    : fixed-width windows with overlap. Structure-blind.
Strategy B  semantic : boundaries fall where consecutive sentences stop being
                       about the same thing, measured by embedding distance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from src.clean import Block

# --------------------------------------------------------------- token sizing

def estimate_tokens(text: str) -> int:
    """
    Token count without a hard tiktoken dependency.

    On Colab tiktoken is available and used for accuracy; offline we fall back
    to the ~4-chars-per-token heuristic, which is close enough for sizing
    decisions and keeps this module importable anywhere.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ------------------------------------------------------------------- chunk

@dataclass
class Chunk:
    text: str
    chunk_id: str
    strategy: str
    section: str
    company: str
    filing_date: str
    kind: str                      # "text" | "table" | "mixed"
    source_url: str = ""
    n_tokens: int = 0
    block_orders: List[int] = field(default_factory=list)
    authority: str = ""
    doc_title: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    # Optional provenance tiers, used when a corpus mixes sources of differing
    # authority. Unused for SEC filings, where `authority` stays empty and the
    # citation falls back to company + date + section.
    TIER_LABEL = {
        "regulation": "regulation",
        "contract": "binding contract",
        "commitment": "voluntary commitment",
    }

    def citation(self) -> str:
        """
        A human-checkable citation.

        For filings: "[NVIDIA 2025-02-26, Item 7 - Management's Discussion]".
        Where a corpus mixes sources of differing authority, the tier is
        appended so a reader can weigh the claim.
        """
        who = self.company or self.doc_title or "Source"
        if self.filing_date:
            who = f"{who} {self.filing_date}"
        tier = self.TIER_LABEL.get(self.authority, "")
        parts = [who]
        if self.section and self.section != self.doc_title:
            parts.append(self.section)
        base = ", ".join(parts)
        return f"[{base}" + (f" ({tier})" if tier else "") + "]"


# ---------------------------------------------------------- sentence splitting

_ABBREV_SET = {
    "inc", "corp", "co", "ltd", "llc", "llp", "plc", "u.s", "u.k", "e.u",
    "mr", "mrs", "ms", "dr", "prof", "st", "no", "nos", "vs", "etc", "al",
    "jr", "sr", "fig", "approx", "sec", "art", "ch", "vol", "dept", "est",
    "cf", "ave", "gen", "adj", "min", "max", "yr", "mo", "qtr", "fy",
}

# A candidate boundary: sentence punctuation followed by whitespace and then
# something that could start a sentence. Requiring the whitespace already
# rules out decimals ("$1.5 billion", "12.4%") for free.
_CAND_RE = re.compile(r"([.!?])\s+(?=[\"\'(\[]?[A-Z0-9])")
_LAST_TOKEN_RE = re.compile(r"[\s(\[\"\']")


def split_sentences(text: str) -> List[str]:
    """
    Conservative sentence splitter tuned for filing prose.

    Python's `re` only supports fixed-width lookbehind, so instead of one
    heroic regex we find candidate boundaries and then reject the ones that
    sit after an abbreviation ("Inc.", "U.S."), a single initial ("J."), or a
    list number ("3."). Over-splitting is the failure mode that hurts semantic
    chunking most, so the bias here is toward splitting less.
    """
    if not text:
        return []
    out: List[str] = []
    start = 0
    for m in _CAND_RE.finditer(text):
        punct_at = m.start(1)
        prev = text[max(0, punct_at - 14): punct_at]
        tok = _LAST_TOKEN_RE.split(prev)[-1].lower()
        if tok in _ABBREV_SET:
            continue
        if re.fullmatch(r"[a-z]", tok):          # "J. Smith"
            continue
        if m.group(1) == "." and re.fullmatch(r"\d+", tok):   # "3. Item"
            continue
        piece = text[start: m.end(1)].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


# ------------------------------------------------------- strategy A: fixed

def chunk_fixed(
    blocks: Sequence[Block],
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    atomic_tables: bool = False,
    strategy_name: str = "fixed",
) -> List[Chunk]:
    """
    Structure-blind baseline.

    The document is linearised into one character stream and cut into windows
    of `chunk_tokens` with `overlap_tokens` of carry-over. This is the control
    condition: it is what you get when you reach for RecursiveCharacterText-
    Splitter without thinking about the corpus.
    """
    chunks: List[Chunk] = []
    # ~4 chars per token; we size in chars and report tokens.
    win = chunk_tokens * 4
    ov = overlap_tokens * 4

    def emit(text: str, meta: Block, orders: List[int], kind: str):
        text = text.strip()
        if not text:
            return
        chunks.append(Chunk(
            text=text,
            chunk_id=f"{strategy_name}-{len(chunks):05d}",
            strategy=strategy_name,
            section=meta.section, company=meta.company,
            filing_date=meta.filing_date, source_url=meta.source_url,
            kind=kind, n_tokens=estimate_tokens(text), block_orders=orders,
            authority=meta.authority, doc_title=meta.doc_title,
        ))

    if atomic_tables:
        # Tables bypass the window entirely; prose is windowed between them.
        buf, buf_orders, buf_meta = "", [], None
        def flush():
            nonlocal buf, buf_orders, buf_meta
            if buf.strip() and buf_meta is not None:
                for piece, orders in _window(buf, buf_orders, win, ov):
                    emit(piece, buf_meta, orders, "text")
            buf, buf_orders, buf_meta = "", [], None

        for b in blocks:
            if b.kind == "table":
                flush()
                emit(b.text, b, [b.order], "table")
            else:
                if buf_meta is None or b.section != buf_meta.section:
                    flush()
                    buf_meta = b
                buf += ("\n" if buf else "") + b.text
                buf_orders.append(b.order)
        flush()
        return chunks

    # Fully naive: one stream, section metadata taken from the block the
    # window starts in. Tables get cut mid-row exactly as they would in a
    # careless pipeline -- that is the point of the baseline.
    stream, index = "", []          # index[i] = block order for char i
    for b in blocks:
        piece = (("\n" if stream else "") + b.text)
        stream += piece
        index.extend([b.order] * len(piece))

    by_order = {b.order: b for b in blocks}
    step = max(1, win - ov)
    for start in range(0, max(1, len(stream)), step):
        piece = stream[start:start + win]
        if not piece.strip():
            continue
        orders = sorted(set(index[start:start + win]))
        meta = by_order[orders[0]] if orders else blocks[0]
        kinds = {by_order[o].kind for o in orders if o in by_order}
        kind = "mixed" if len(kinds) > 1 else (kinds.pop() if kinds else "text")
        emit(piece, meta, orders, kind)
        if start + win >= len(stream):
            break
    return chunks


def _window(text: str, orders: List[int], win: int, ov: int):
    step = max(1, win - ov)
    if len(text) <= win:
        yield text, orders
        return
    for s in range(0, len(text), step):
        piece = text[s:s + win]
        if piece.strip():
            yield piece, orders
        if s + win >= len(text):
            break


# ---------------------------------------------------- strategy B: semantic

def chunk_semantic(
    blocks: Sequence[Block],
    embed_fn: Callable[[List[str]], np.ndarray],
    breakpoint_percentile: float = 90.0,
    max_tokens: int = 768,
    min_tokens: int = 64,
    atomic_tables: bool = True,
    strategy_name: str = "semantic",
) -> List[Chunk]:
    """
    Boundaries where meaning shifts.

    For each run of prose inside a section we embed every sentence, take the
    cosine distance between consecutive sentences, and cut wherever that
    distance exceeds the Nth percentile of distances in the document. High
    distance == the topic moved on, which is where a human would break.

    `max_tokens` caps runaway chunks (a long uniform passage produces no
    breakpoints); `min_tokens` merges slivers forward so we do not embed
    one-line fragments that retrieve poorly.
    """
    # Group consecutive non-table blocks by section.
    groups: List[List[Block]] = []
    cur: List[Block] = []
    tables: List[Block] = []

    ordered: List[tuple] = []   # ("table", Block) | ("group", [Block])
    for b in blocks:
        if b.kind == "table" and atomic_tables:
            if cur:
                ordered.append(("group", cur)); cur = []
            ordered.append(("table", b))
        else:
            if cur and b.section != cur[-1].section:
                ordered.append(("group", cur)); cur = []
            cur.append(b)
    if cur:
        ordered.append(("group", cur))

    # Collect every sentence up front so we embed in ONE batched call.
    sent_map: List[tuple] = []      # (group_idx, sentence)
    group_sents: Dict[int, List[str]] = {}
    for gi, (tag, payload) in enumerate(ordered):
        if tag != "group":
            continue
        text = "\n".join(b.text for b in payload)
        sents = split_sentences(text) or [text]
        group_sents[gi] = sents
        for s in sents:
            sent_map.append((gi, s))

    all_sents = [s for _, s in sent_map]
    if all_sents:
        vecs = np.asarray(embed_fn(all_sents), dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    else:
        vecs = np.zeros((0, 1), dtype=np.float32)

    # Per-group consecutive distances, pooled to pick one global threshold.
    offsets: Dict[int, int] = {}
    o = 0
    for gi in sorted(group_sents):
        offsets[gi] = o
        o += len(group_sents[gi])

    all_d: List[float] = []
    group_d: Dict[int, np.ndarray] = {}
    for gi, sents in group_sents.items():
        if len(sents) < 2:
            group_d[gi] = np.zeros(0, dtype=np.float32)
            continue
        v = vecs[offsets[gi]: offsets[gi] + len(sents)]
        sims = np.sum(v[:-1] * v[1:], axis=1)
        d = 1.0 - sims
        group_d[gi] = d
        all_d.extend(d.tolist())

    threshold = float(np.percentile(all_d, breakpoint_percentile)) if all_d else 1.0

    chunks: List[Chunk] = []

    def emit(text: str, meta: Block, orders: List[int], kind: str):
        text = text.strip()
        if not text:
            return
        chunks.append(Chunk(
            text=text, chunk_id=f"{strategy_name}-{len(chunks):05d}",
            strategy=strategy_name, section=meta.section, company=meta.company,
            filing_date=meta.filing_date, source_url=meta.source_url,
            kind=kind, n_tokens=estimate_tokens(text), block_orders=orders,
            authority=meta.authority, doc_title=meta.doc_title,
        ))

    for gi, (tag, payload) in enumerate(ordered):
        if tag == "table":
            emit(payload.text, payload, [payload.order], "table")
            continue

        sents = group_sents[gi]
        meta = payload[0]
        orders = [b.order for b in payload]
        d = group_d[gi]

        pieces, buf = [], ""
        for i, s in enumerate(sents):
            cand = (buf + " " + s).strip() if buf else s
            over_cap = estimate_tokens(cand) > max_tokens
            is_break = (i < len(d) and d[i] > threshold)
            if over_cap and buf:
                pieces.append(buf); buf = s
            else:
                buf = cand
                if is_break and estimate_tokens(buf) >= min_tokens:
                    pieces.append(buf); buf = ""
        if buf:
            pieces.append(buf)

        # Merge any sliver forward so nothing tiny reaches the index.
        merged: List[str] = []
        for p in pieces:
            if merged and estimate_tokens(p) < min_tokens:
                merged[-1] = merged[-1] + " " + p
            else:
                merged.append(p)
        for p in merged:
            emit(p, meta, orders, "text")

    return chunks


# ------------------------------------------------------------------ reporting

def chunk_stats(chunks: Sequence[Chunk]) -> Dict:
    toks = [c.n_tokens for c in chunks] or [0]
    tables = [c for c in chunks if c.kind == "table"]
    return {
        "strategy": chunks[0].strategy if chunks else "-",
        "n_chunks": len(chunks),
        "tokens_mean": round(float(np.mean(toks)), 1),
        "tokens_median": float(np.median(toks)),
        "tokens_min": int(np.min(toks)),
        "tokens_max": int(np.max(toks)),
        "table_chunks": len(tables),
        "sections_covered": len({c.section for c in chunks}),
    }
