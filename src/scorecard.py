"""
The evaluation deliverable, kept deliberately small.

The Week 2 sessions were explicit that evaluation is bonus credit and that a
full framework is not wanted -- Ragas, LangSmith and formal metric suites are
Week 4 material. What they asked for is a short table showing the app was
checked, and at least one failure explained as either "it retrieved the wrong
evidence" or "it wrote a bad answer from good evidence".

That distinction is the whole point. The four-quadrant picture from Session 2:

                      generation good        generation bad
    retrieval good    what you want          prompt problem
    retrieval bad     should have refused    everything is wrong

The dangerous box is bad retrieval + a confident answer -- the model found
nothing and answered anyway. When retrieval comes back empty, the correct
behaviour is to say so. A system that guesses instead has a GENERATION
problem, not a retrieval one, and the fixes are completely different.

This module is ~80 lines on purpose. It should be readable by the person
presenting it.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


# Phrases that indicate the model declined rather than guessed.
_DECLINED = (
    "not in", "does not appear", "cannot be answered", "can't be answered",
    "no information", "not disclosed", "not available", "unable to answer",
    "not found in", "do not contain", "does not contain", "not present in",
    "cannot determine", "i don't have", "i do not have", "not covered",
)

_ASKED_BACK = (
    "which company", "which of the", "could you specify", "please specify",
    "which filing", "which fiscal", "which period", "did you mean", "clarify",
)


def declined(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in _DECLINED)


def asked_back(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in _ASKED_BACK) and "?" in a


def diagnose(r: Dict) -> str:
    """
    Why this question went the way it did.

    `r` needs: question, retrieved_ok ("yes"|"partly"|"no"), answer,
    n_citations, and optionally should_refuse / should_ask.
    """
    a = r.get("answer", "")
    got_out = declined(a) or asked_back(a)

    if r.get("should_refuse") or r.get("should_ask"):
        return ("correct — declined instead of guessing" if got_out
                else "GENERATION — guessed when it should have declined")

    if r.get("retrieved_ok") == "no":
        return ("correct — no evidence, and it said so" if got_out
                else "RETRIEVAL — wrong passages came back, and it answered anyway")

    if r.get("retrieved_ok") == "partly":
        return "RETRIEVAL — only some of the evidence came back"

    # Evidence was there. Anything wrong now is the generator's doing.
    if got_out:
        return "GENERATION — had the right passages and declined anyway"
    if not r.get("n_citations"):
        return "GENERATION — answered without citing a source"
    return "correct"


def _evidence_cell(r: Dict) -> str:
    if r.get("should_refuse"):
        return "No evidence exists"
    if r.get("should_ask"):
        return "n/a — question underspecified"
    return {"yes": "Yes", "partly": "Partly", "no": "No"}.get(
        r.get("retrieved_ok", ""), "not checked")


def _grounded_cell(r: Dict) -> str:
    a = r.get("answer", "")
    if r.get("should_refuse") or r.get("should_ask"):
        return "Yes" if (declined(a) or asked_back(a)) else "No"
    if declined(a) or asked_back(a):
        return "n/a — declined"
    return "Yes" if r.get("n_citations") else "No — uncited"


def scorecard(results: Sequence[Dict], max_q: int = 62) -> str:
    """The four-column table the sessions describe as the bonus deliverable."""
    out = ["| Test question | Right evidence retrieved? | "
           "Answer grounded and cited? | What happened |",
           "|---|---|---|---|"]
    for r in results:
        q = r.get("question", "")
        q = q if len(q) <= max_q else q[:max_q - 1] + "…"
        out.append(f"| {q} | {_evidence_cell(r)} | {_grounded_cell(r)} | {diagnose(r)} |")
    return "\n".join(out)


def failures(results: Sequence[Dict]) -> str:
    """
    The failures, split by cause, with the fix each one points at.

    Including at least one of these, explained, is what the sessions said
    makes the evaluation section look strong.
    """
    retrieval, generation = [], []
    for r in results:
        d = diagnose(r)
        if d.startswith("RETRIEVAL"):
            retrieval.append((r, d))
        elif d.startswith("GENERATION"):
            generation.append((r, d))

    if not retrieval and not generation:
        return ("No failures recorded. Worth double-checking the questions are "
                "hard enough — an easy test set also produces no failures.")

    out = []
    if retrieval:
        out.append(f"RETRIEVAL FAULTS ({len(retrieval)})")
        out.append("Fix the retrieval side: chunk size, keeping tables whole, "
                   "hybrid search, metadata, reranking. Do not touch the prompt.")
        for r, d in retrieval:
            out.append(f"\n  {r.get('question','')}\n    {d}")
        out.append("")
    if generation:
        out.append(f"GENERATION FAULTS ({len(generation)})")
        out.append("Fix the generation side: the answer prompt, the instruction "
                   "to cite, the instruction to decline when evidence is missing.")
        for r, d in generation:
            out.append(f"\n  {r.get('question','')}\n    {d}")
        out.append("")
    out.append(f"{len(retrieval)} retrieval · {len(generation)} generation")
    return "\n".join(out)
