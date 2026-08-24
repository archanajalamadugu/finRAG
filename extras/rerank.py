"""
Reranking: the second look.

What a reranker is actually for
-------------------------------
Retrieval has to score a query against every chunk in the corpus, so it can
only afford to compare two vectors that were computed independently -- the
chunk never sees the question. A cross-encoder reads the query and the chunk
*together* and scores the pair, which is far more accurate and far too slow
to run over 5,000 chunks. So the pipeline uses each for what it is good at:
retrieval casts a wide cheap net (top-20 per retriever), reranking makes an
expensive careful decision about the survivors (top-5).

That framing is also why the reranking analysis has to report latency. A
reranker that adds four points of Recall@5 and three seconds of p95 is not
obviously a win for an analyst waiting on an answer, and a report that only
shows the quality column is hiding half the trade-off.

Three implementations, one interface
-------------------------------------
`CrossEncoderReranker`  the real thing; a local ms-marco cross-encoder.
`LLMListwiseReranker`   fallback when the model download fails or there is no
                        GPU. Shows the LLM all candidates at once and asks for
                        an order, which is more robust than scoring each
                        candidate separately -- pointwise LLM scores are
                        notoriously badly calibrated against each other.
`IdentityReranker`      passes retrieval order straight through. This is the
                        control condition for run B2 and it exists so the
                        "with rerank" and "without rerank" paths run the same
                        code with the same timing instrumentation.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .retrieve import Hit


# ------------------------------------------------------------------ timing

@dataclass
class Stage:
    """One timed pipeline stage. Latency is a reported metric, not a vibe."""
    name: str
    seconds: float


class Timer:
    """Accumulates per-stage wall-clock so a run can report where time went."""

    def __init__(self):
        self.stages: List[Stage] = []

    def time(self, name: str, fn: Callable, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.stages.append(Stage(name, time.perf_counter() - t0))
        return out

    def total(self) -> float:
        return sum(s.seconds for s in self.stages)

    def as_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for s in self.stages:
            d[s.name] = d.get(s.name, 0.0) + s.seconds
        d["total"] = self.total()
        return d


# --------------------------------------------------------------- rerankers

class IdentityReranker:
    """No-op. The control condition."""
    name = "none"

    def rerank(self, query: str, chunks: Sequence[dict], hits: Sequence[Hit],
               top_k: int = 5) -> List[Hit]:
        out = []
        for r, h in enumerate(hits[:top_k]):
            out.append(Hit(idx=h.idx, score=h.score, source=h.source,
                           rank=r + 1, components=dict(h.components)))
        return out


class CrossEncoderReranker:
    """
    Local cross-encoder over (query, chunk) pairs.

    `ms-marco-MiniLM-L-6-v2` is small enough to run on a Colab CPU in a couple
    of seconds for 20 candidates and is the standard baseline reranker, which
    makes the measured improvement comparable to published numbers rather than
    being a property of some model nobody else has.

    Chunk text is truncated to `max_chars` before scoring: the model's window
    is 512 tokens and a 768-token table chunk would otherwise be silently cut
    at whatever the tokenizer reached first. Truncating deliberately at least
    keeps the head of the table -- the header row and the first data rows --
    which is where the answer usually is.
    """
    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_chars: int = 1600, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512, device=device)
        self.model_name = model_name
        self.max_chars = max_chars

    def rerank(self, query: str, chunks: Sequence[dict], hits: Sequence[Hit],
               top_k: int = 5) -> List[Hit]:
        if not hits:
            return []
        pairs = [(query, chunks[h.idx]["text"][:self.max_chars]) for h in hits]
        scores = self.model.predict(pairs)
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        out = []
        for r, i in enumerate(order[:top_k]):
            h = hits[i]
            comp = dict(h.components)
            comp["pre_rerank"] = h.rank
            out.append(Hit(idx=h.idx, score=float(scores[i]),
                           source="rerank", rank=r + 1, components=comp))
        return out


_ORDER_RE = re.compile(r"\d+")


class LLMListwiseReranker:
    """
    Fallback reranker: show the model every candidate, ask for an ordering.

    Listwise rather than pointwise on purpose. Asking an LLM "score this
    passage 0-10" gives numbers that are not comparable across calls, so the
    resulting order is mostly noise. Asking it to rank a visible list is a
    comparison it can actually make.

    Robustness matters more than elegance here, because this runs when
    something else has already failed. The parser accepts a bare list of
    numbers in any punctuation, ignores out-of-range and duplicate indices,
    and appends anything the model omitted in its original retrieval order --
    so a mangled response degrades to "retrieval order" rather than throwing.
    """
    name = "llm-listwise"

    PROMPT = (
        "You are ranking passages from SEC 10-K filings by how well each one "
        "answers a specific question.\n\n"
        "QUESTION: {query}\n\n"
        "PASSAGES:\n{passages}\n\n"
        "Rank the passage numbers from most to least useful for answering the "
        "question. A passage containing the specific figure, segment or "
        "statement asked about outranks one that merely discusses the topic.\n"
        "Reply with ONLY the numbers, most useful first, comma-separated. "
        "No explanation.\n"
    )

    def __init__(self, client, model: str, max_chars: int = 900,
                 max_candidates: int = 20):
        self.client, self.model = client, model
        self.max_chars, self.max_candidates = max_chars, max_candidates

    def rerank(self, query: str, chunks: Sequence[dict], hits: Sequence[Hit],
               top_k: int = 5) -> List[Hit]:
        if not hits:
            return []
        hits = list(hits)[:self.max_candidates]
        passages = "\n\n".join(
            f"[{i}] ({chunks[h.idx].get('company','?')} · "
            f"{chunks[h.idx].get('section','?')})\n"
            f"{chunks[h.idx]['text'][:self.max_chars]}"
            for i, h in enumerate(hits))
        try:
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=120, temperature=0,
                messages=[{"role": "user",
                           "content": self.PROMPT.format(query=query, passages=passages)}])
            raw = r.choices[0].message.content or ""
        except Exception:
            raw = ""

        seen, order = set(), []
        for m in _ORDER_RE.findall(raw):
            i = int(m)
            if 0 <= i < len(hits) and i not in seen:
                seen.add(i)
                order.append(i)
        # Anything the model dropped keeps its retrieval position, at the back.
        order += [i for i in range(len(hits)) if i not in seen]

        out = []
        for r, i in enumerate(order[:top_k]):
            h = hits[i]
            comp = dict(h.components)
            comp["pre_rerank"] = h.rank
            out.append(Hit(idx=h.idx, score=1.0 / (r + 1), source="rerank",
                           rank=r + 1, components=comp))
        return out


# --------------------------------------------------------------- diagnostics

def rank_movement(before: Sequence[Hit], after: Sequence[Hit]) -> Dict:
    """
    What the reranker actually did.

    A rerank step that changes nothing and a rerank step that helps produce
    the same top-line metric when the retriever was already right. Reporting
    movement separates "the reranker was not needed here" from "the reranker
    did nothing" -- and a reranker that reorders heavily while metrics stay
    flat is a warning, not a success.
    """
    before_ids = [h.idx for h in before]
    after_ids = [h.idx for h in after]
    promoted = [i for i in after_ids if i in before_ids
                and after_ids.index(i) < before_ids.index(i)]
    entered = [i for i in after_ids if i not in before_ids[:len(after_ids)]]
    return {
        "n_before": len(before_ids),
        "n_after": len(after_ids),
        "top1_changed": bool(before_ids and after_ids and before_ids[0] != after_ids[0]),
        "n_promoted": len(promoted),
        "n_newly_in_topk": len(entered),
        "unchanged": before_ids[:len(after_ids)] == after_ids,
    }
