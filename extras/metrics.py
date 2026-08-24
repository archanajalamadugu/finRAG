"""
Evaluation metrics.

This module is deliberately framework-free. Recall@k, MRR and nDCG are
arithmetic, not a library, and writing them out means the numbers in the
report can be checked by hand against the eval file -- which is the whole
point of reporting them. RAGAS or a similar harness can sit on top of this
for the generation-side metrics; the retrieval side does not need it.

Three families of metric, because the eval set has three kinds of question:

  retrieval    Did the right chunk come back, and how high? Recall@k, MRR@k,
               nDCG@k, plus source diversity for cross-company questions.
  answer       Is the generated answer right and supported? Numeric exact
               match for the table lookups, LLM-judged faithfulness for prose.
  behaviour    Did the system refuse when it should have, and ask for
               clarification when the question was ambiguous? Scored
               separately so correct restraint is never counted as failure.

The third family is the one most projects skip, and it is where a financial
RAG system does its most important work. An analyst who gets a confident
fabricated number is worse off than one who gets "not in this filing".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set


# ------------------------------------------------------------------ retrieval

def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """Fraction of the labelled relevant chunks that appear in the top k.

    The metric to optimise for risk_synthesis questions, where the answer is
    assembled from several passages and missing one makes the answer wrong.
    """
    rel = set(relevant)
    if not rel:
        return float("nan")     # unlabelled: excluded from the mean, not scored 0
    return len(rel & set(retrieved[:k])) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    rel = set(relevant)
    if not rel or k == 0:
        return float("nan")
    return len(rel & set(retrieved[:k])) / min(k, len(retrieved)) if retrieved else 0.0


def mrr_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Reciprocal rank of the FIRST relevant chunk.

    The metric that matters for table_lookup: one chunk holds the number, and
    whether it arrives at rank 1 or rank 8 decides whether the generator sees
    it inside a truncated context.
    """
    rel = set(relevant)
    if not rel:
        return float("nan")
    for i, cid in enumerate(retrieved[:k]):
        if cid in rel:
            return 1.0 / (i + 1)
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 10,
              grades: Optional[Dict[str, float]] = None) -> float:
    """
    Rank-aware, and unlike MRR it rewards getting the *second* and *third*
    relevant chunk high too -- which is what separates a good cross-company
    answer from one built on a single filing.

    `grades` allows graded relevance (2 = contains the answer, 1 = supporting
    context). With no grades supplied every relevant chunk counts 1.
    """
    rel = set(relevant)
    if not rel:
        return float("nan")
    g = grades or {}
    got = [g.get(cid, 1.0) if cid in rel else 0.0 for cid in retrieved[:k]]
    ideal = sorted((g.get(cid, 1.0) for cid in rel), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(got) / idcg if idcg > 0 else 0.0


# -------------------------------------------------------------------- numbers

# "$47,525 million", "47.5 billion", "(1,234)", "12.4%"
_NUM_RE = re.compile(
    r"\(?\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*\)?\s*"
    r"(billion|bn|million|mm|m\b|thousand|k\b|%)?",
    re.IGNORECASE)

_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mm": 1e6, "m": 1e6,
          "thousand": 1e3, "k": 1e3}


def extract_numbers(text: str) -> List[float]:
    """
    Pull comparable magnitudes out of free text.

    Financial answers state the same quantity many ways: a filing table says
    `60,922` under a "$ in millions" header, the model says "$60.9 billion",
    and a naive string comparison calls that wrong. Every number is therefore
    scaled to absolute units before comparison. Parenthesised values are
    negated, since accounting statements write losses that way and a sign
    error is a real error worth catching.
    """
    out: List[float] = []
    for m in _NUM_RE.finditer(text or ""):
        raw, unit = m.group(1), (m.group(2) or "").lower().strip()
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit == "%":
            out.append(v)
            continue
        v *= _SCALE.get(unit, 1.0)
        span = m.group(0)
        if span.strip().startswith("(") and span.strip().endswith(")"):
            v = -v
        out.append(v)
    return out


def numeric_match(answer: str, expected: float, rel_tol: float = 0.01,
                  allow_scale_slip: bool = True) -> bool:
    """
    Did the answer contain the expected figure?

    Tolerance is relative, not absolute, because "approximately $60.9 billion"
    is a correct answer to a question whose ground truth is 60,922 million and
    an absolute epsilon cannot express that.

    `allow_scale_slip` also accepts the value off by exactly 1,000x. That is
    not leniency -- it is a deliberate probe. A model reading a table under a
    "$ in millions" header and reporting the bare number is a *specific*
    failure of table-header retention, and counting it separately tells us
    whether the chunking strategy is dropping table headers. Runs report both
    `numeric_exact` and `numeric_scale_slip` so the two never blur together.
    """
    got = extract_numbers(answer)
    if not got:
        return False
    for v in got:
        if expected == 0:
            if abs(v) < 1e-9:
                return True
            continue
        if abs(v - expected) / abs(expected) <= rel_tol:
            return True
        if allow_scale_slip:
            for f in (1e3, 1e-3, 1e6, 1e-6):
                if abs(v * f - expected) / abs(expected) <= rel_tol:
                    return True
    return False


def numeric_verdict(answer: str, expected: float, rel_tol: float = 0.01) -> str:
    """"exact" | "scale_slip" | "wrong" | "no_number"."""
    got = extract_numbers(answer)
    if not got:
        return "no_number"
    if numeric_match(answer, expected, rel_tol, allow_scale_slip=False):
        return "exact"
    if numeric_match(answer, expected, rel_tol, allow_scale_slip=True):
        return "scale_slip"
    return "wrong"


# ------------------------------------------------------------------ behaviour

_REFUSAL_MARKERS = (
    "not in", "does not appear", "cannot be answered", "can't be answered",
    "not contained", "no information", "not disclosed", "not available",
    "unable to answer", "not found in", "outside the scope", "do not contain",
    "does not contain", "not present in", "insufficient", "cannot determine",
    "not something", "not covered", "i don't have", "i do not have",
)

_CLARIFY_MARKERS = (
    "which company", "which of the", "could you specify", "can you specify",
    "please specify", "which filing", "which fiscal", "which period",
    "did you mean", "clarify", "which one", "specify which",
)


def looks_like_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in _REFUSAL_MARKERS)


def looks_like_clarification(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in _CLARIFY_MARKERS) and "?" in a


def behaviour_verdict(answer: str, expect_refusal: bool = False,
                      expect_clarification: bool = False) -> Dict:
    """
    Score the ambiguous and unanswerable cases on what the system *did*.

    Kept apart from the accuracy metrics on purpose. If a refusal were
    averaged into "answer accuracy" it would read as a failure, and the
    system would be rewarded for guessing. Here, guessing is the failure.
    """
    refused = looks_like_refusal(answer)
    clarified = looks_like_clarification(answer)
    if expect_refusal:
        return {"expected": "refuse", "got": "refuse" if refused else
                ("clarify" if clarified else "answered"), "correct": refused}
    if expect_clarification:
        ok = clarified or refused
        return {"expected": "clarify", "got": "clarify" if clarified else
                ("refuse" if refused else "answered"), "correct": ok}
    # A normal question: refusing it is the failure mode to watch for.
    return {"expected": "answer",
            "got": "refuse" if refused else "answered",
            "correct": not refused}


def refusal_precision(records: Sequence[Dict]) -> Dict:
    """
    Refusal behaviour as a confusion matrix, not a single number.

    Over-refusal and under-refusal are different problems with different
    fixes, and one accuracy figure hides which one you have.
    """
    tp = fp = fn = tn = 0
    for r in records:
        should = bool(r.get("expect_refusal") or r.get("expect_clarification"))
        did = r.get("got") in ("refuse", "clarify")
        if should and did:
            tp += 1
        elif not should and did:
            fp += 1           # refused a question it could have answered
        elif should and not did:
            fn += 1           # answered something it should not have
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"refusal_precision": prec, "refusal_recall": rec,
            "over_refusals": fp, "missed_refusals": fn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------- aggregation

def _mean(xs: Iterable[float]) -> float:
    """Mean that skips NaN, so unlabelled questions abstain rather than score 0."""
    vals = [x for x in xs if x == x]
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)


@dataclass
class RunResult:
    """Everything one row of the experiment matrix produced."""
    run_id: str
    config: Dict = field(default_factory=dict)
    per_question: List[Dict] = field(default_factory=list)

    def summary(self, k: int = 5) -> Dict:
        qs = self.per_question
        scored = [r for r in qs if not (r.get("expect_refusal")
                                        or r.get("expect_clarification"))]
        behaviour = [r for r in qs if (r.get("expect_refusal")
                                       or r.get("expect_clarification"))]
        lat = [r["latency_total"] for r in qs if r.get("latency_total") is not None]
        numeric = [r for r in scored if r.get("numeric_verdict")]

        out = {
            "run_id": self.run_id,
            "n_questions": len(qs),
            f"recall@{k}": _mean(r.get("recall", float("nan")) for r in scored),
            "mrr@10": _mean(r.get("mrr", float("nan")) for r in scored),
            "ndcg@10": _mean(r.get("ndcg", float("nan")) for r in scored),
            "faithfulness": _mean(r.get("faithfulness", float("nan")) for r in scored),
            "source_diversity_ok": _mean(
                r.get("diversity_ok", float("nan")) for r in scored),
            "numeric_exact": (
                sum(1 for r in numeric if r["numeric_verdict"] == "exact") / len(numeric)
                if numeric else float("nan")),
            "numeric_scale_slip": (
                sum(1 for r in numeric if r["numeric_verdict"] == "scale_slip") / len(numeric)
                if numeric else float("nan")),
            "behaviour_correct": (
                sum(1 for r in behaviour if r.get("behaviour_correct")) / len(behaviour)
                if behaviour else float("nan")),
            # How many questions this chunking policy destroyed the evidence
            # for. Belongs next to recall, because it explains recall.
            "evidence_lost": sum(1 for r in scored if r.get("evidence_lost")),
            "latency_p50": percentile(lat, 50),
            "latency_p95": percentile(lat, 95),
        }
        out.update(refusal_precision(qs))
        out.update({f"cfg_{k2}": v for k2, v in self.config.items()})
        return out


COMPARE_COLS = ["run_id", "recall@5", "mrr@10", "ndcg@10", "evidence_lost",
                "numeric_exact", "faithfulness", "behaviour_correct",
                "latency_p50", "latency_p95"]


def comparison_table(runs: Sequence[RunResult], cols: Sequence[str] = None) -> str:
    """Markdown table, ready to paste into the write-up."""
    cols = list(cols or COMPARE_COLS)
    rows = [r.summary() for r in runs]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append("—" if v != v else f"{v:.3f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


# ------------------------------------------------------- the plain-English view

def diagnose(rec: Dict) -> str:
    """
    Retrieval fault, or generation fault?

    This is the distinction the Week 2 sessions single out as the point of
    evaluating a RAG app at all: telling apart "the app found the wrong
    evidence" from "the app wrote a bad answer from good evidence". The two
    have different fixes -- chunking, metadata, hybrid search and reranking on
    one side; the answer prompt on the other -- so a report that cannot
    separate them cannot tell you what to change next.
    """
    if rec.get("evidence_lost"):
        return "retrieval — chunking destroyed the evidence"
    r = rec.get("recall")
    faithful = rec.get("faithfulness")
    cited = rec.get("n_cited", 0)

    if rec.get("expect_refusal") or rec.get("expect_clarification"):
        return ("correct restraint" if rec.get("behaviour_correct")
                else "generation — guessed instead of declining")

    if r == 0.0:
        return "retrieval — right passage never came back"
    if r is not None and r == r and r < 1.0:
        base = "retrieval — only part of the evidence came back"
        if faithful is not None and faithful == faithful and faithful < 0.9:
            return base + ", and the answer overreached on what it had"
        return base
    # Evidence was there. Anything wrong now is the generator's doing.
    if faithful is not None and faithful == faithful and faithful < 0.9:
        return "generation — had the right evidence, made unsupported claims"
    if rec.get("numeric_verdict") == "scale_slip":
        return "generation — right figure, dropped the table's units"
    if rec.get("numeric_verdict") in ("wrong", "no_number"):
        return "generation — had the right evidence, reported the wrong figure"
    if not cited and rec.get("answer_mode") == "answer":
        return "generation — answered without citing anything"
    return "correct"


def _evidence_cell(rec: Dict) -> str:
    if rec.get("expect_refusal"):
        return "No evidence exists"
    if rec.get("expect_clarification"):
        return "n/a — question underspecified"
    if rec.get("evidence_lost"):
        return "No"
    r = rec.get("recall")
    if r is None or r != r:
        return "unlabelled"
    return "Yes" if r >= 1.0 else ("Partly" if r > 0 else "No")


def _grounded_cell(rec: Dict) -> str:
    if rec.get("expect_refusal") or rec.get("expect_clarification"):
        return "Yes" if rec.get("behaviour_correct") else "No"
    f = rec.get("faithfulness")
    if f is None or f != f:
        return "not judged"
    if f < 0.9:
        return "No"
    return "Yes" if rec.get("n_cited", 0) else "Yes, but uncited"


def simple_scorecard(run: RunResult, max_q_chars: int = 62) -> str:
    """
    The four-column table the Week 2 sessions describe as the whole bonus
    deliverable: question, was the right evidence retrieved, was the answer
    grounded and cited, and what happened in plain English.

    Everything else in this module is supporting detail for the Project 2
    requirements -- comparing two chunking strategies and measuring what
    reranking bought. This is the table a human reads. It goes first in the
    write-up and on screen in the demo; the metric tables go in an appendix.
    """
    lines = ["| Test question | Right evidence retrieved? | "
             "Answer grounded and cited? | What happened |",
             "|---|---|---|---|"]
    for rec in run.per_question:
        q = rec.get("question", "")
        q = q if len(q) <= max_q_chars else q[:max_q_chars - 1] + "…"
        lines.append(f"| {q} | {_evidence_cell(rec)} | "
                     f"{_grounded_cell(rec)} | {diagnose(rec)} |")
    return "\n".join(lines)


def failure_summary(run: RunResult, limit: int = 5) -> str:
    """
    The failures, split by cause, with the fix each one points at.

    "Include at least one failure and explain whether it failed because the
    wrong chunk was retrieved or the answer used good evidence poorly" is the
    single thing the sessions say makes this deliverable look strong. This
    groups them so the write-up can say which kind of problem dominates.
    """
    buckets: Dict[str, List[Dict]] = {"retrieval": [], "generation": [], "other": []}
    for rec in run.per_question:
        d = diagnose(rec)
        if d.startswith("correct"):
            continue
        key = "retrieval" if d.startswith("retrieval") else (
            "generation" if d.startswith("generation") else "other")
        buckets[key].append((rec, d))

    fixes = {
        "retrieval": "Fix on the retrieval side: chunk boundaries, table "
                     "atomicity, metadata filters, hybrid weighting, reranking.",
        "generation": "Fix on the generation side: the answer prompt, the "
                      "units instruction, the citation requirement.",
        "other": "Fix the fallback prompt — the app is not declining cleanly.",
    }
    out = [f"FAILURE ANALYSIS · {run.run_id}", "=" * 58]
    total = sum(len(v) for v in buckets.values())
    if not total:
        out.append("\nNo failures recorded. Confirm the eval set is labelled — "
                   "an unlabelled set also produces no failures.")
        return "\n".join(out)
    for kind, items in buckets.items():
        if not items:
            continue
        out.append(f"\n{kind.upper()} FAULTS ({len(items)})")
        out.append(fixes[kind])
        for rec, d in items[:limit]:
            out.append(f"\n  {rec['id']} [{rec.get('type','')}]  {rec.get('question','')}")
            out.append(f"    what happened: {d}")
            if rec.get("unsupported_claims"):
                out.append(f"    unsupported:   {rec['unsupported_claims'][:2]}")
            if rec.get("answer"):
                out.append(f"    answer:        "
                           f"{rec['answer'][:150].replace(chr(10), ' ')}")
    out.append(f"\n{'-' * 58}")
    out.append(f"{len(buckets['retrieval'])} retrieval · "
               f"{len(buckets['generation'])} generation · "
               f"{len(buckets['other'])} fallback")
    return "\n".join(out)


def delta_table(baseline: RunResult, treatment: RunResult,
                cols: Sequence[str] = None) -> str:
    """
    The two required comparisons are both "B minus A", so render them that way.

    Showing the delta rather than two separate tables is what turns a results
    dump into a finding: the reader sees the size and direction of the effect
    without doing arithmetic, including where the effect is negative.
    """
    cols = list(cols or ["recall@5", "mrr@10", "ndcg@10", "numeric_exact",
                         "faithfulness", "latency_p50", "latency_p95"])
    a, b = baseline.summary(), treatment.summary()
    lines = [f"| metric | {baseline.run_id} | {treatment.run_id} | delta |",
             "|---|---|---|---|"]
    for c in cols:
        va, vb = a.get(c), b.get(c)
        if not isinstance(va, float) or not isinstance(vb, float) \
                or va != va or vb != vb:
            lines.append(f"| {c} | — | — | — |")
            continue
        d = vb - va
        lines.append(f"| {c} | {va:.3f} | {vb:.3f} | {d:+.3f} |")
    return "\n".join(lines)
