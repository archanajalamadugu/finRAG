"""
Answer generation: cited, grounded, and willing to say no.

The refusal path was designed before the answer path, on purpose.

A financial RAG system's worst output is not "I don't know" -- it is a
confident, well-formatted, plausible number that is not in the filing. An
analyst can act on that. So the prompt is built around three exits, and the
system is required to take one of them:

  ANSWER      the retrieved passages contain it -> answer and cite
  CLARIFY     the question does not identify which company or period, and
              several are indexed -> ask, do not pick one
  REFUSE      the passages do not contain it -> say so, and say why if the
              reason is structural ("a 10-K does not carry forward guidance")

The third exit is the one that makes the other two trustworthy, and the
prompt spends more words on it than on formatting because that is where the
model's default behaviour needs the most correcting -- a helpful assistant
wants to be helpful, and here restraint is the helpful act.

Citations are rendered from chunk metadata, not asked for from the model.
A model asked to produce citation strings will produce plausible ones; the
only citation worth printing is one assembled from the record that was
actually retrieved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

SYSTEM = """You answer questions about SEC 10-K filings for equity research analysts.

You will be given numbered passages retrieved from the filings of NVIDIA, AMD,
Intel and Broadcom. Answer ONLY from those passages.

Choose exactly one of three responses.

1. ANSWER — the passages contain what was asked.
   State the figure or finding directly, first sentence.
   Cite every claim with the passage number in square brackets, like [2].
   When you give a number, give it with the units the filing uses. If a table
   header says "$ in millions", the figure 60,922 means $60,922 million — say
   so. Never restate a table figure without its scale.
   If passages from different companies are relevant, address each separately.

2. CLARIFY — the question does not say which company, or which fiscal period,
   and more than one is present in the corpus.
   Ask which one. Do not pick the most likely. Do not answer for all four as a
   hedge. One short question is the entire response.

3. REFUSE — the passages do not contain the answer.
   Say plainly that the filings retrieved do not contain it. Where the reason
   is structural, name it — forward-looking guidance, quarterly results and
   executive compensation detail are not in a 10-K, they are in earnings
   releases, 10-Qs and the DEF 14A proxy respectively.
   Do not substitute a related figure you did find. Do not estimate.
   Do not add "however, based on general knowledge…" — there is no such thing
   here.

Never use knowledge from outside the passages, even when you are confident it
is correct. An unsupported true statement and an unsupported false statement
are indistinguishable to the reader, so both are failures."""

USER_TEMPLATE = """PASSAGES
{passages}

QUESTION
{query}"""


@dataclass
class Answer:
    text: str
    citations: List[str] = field(default_factory=list)
    cited_idx: List[int] = field(default_factory=list)
    used_chunks: List[dict] = field(default_factory=list)
    mode: str = "answer"                # answer | clarify | refuse
    usage: Dict = field(default_factory=dict)


def format_passages(chunks: Sequence[dict], max_chars: int = 1800) -> str:
    """
    Render retrieved chunks for the prompt.

    Each passage is labelled with company, filing date and Item section
    *before* its text. That header does double duty: it lets the model attribute
    a figure to the right filing when four companies are in context, and it
    means a cross-company question cannot be answered without the model having
    seen which passage came from where.
    """
    out = []
    for i, c in enumerate(chunks, start=1):
        head = " · ".join(x for x in (
            c.get("company", ""), c.get("filing_date", ""), c.get("section", "")) if x)
        body = c.get("text", "")
        if len(body) > max_chars:
            # Truncate at the tail. For table chunks the header row and the
            # first data rows are at the top and are what the answer needs.
            body = body[:max_chars] + "\n… [passage truncated]"
        out.append(f"[{i}] {head}\n{body}")
    return "\n\n".join(out)


_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def classify_mode(text: str) -> str:
    """Which of the three exits the model actually took."""
    from extras.metrics import looks_like_clarification, looks_like_refusal
    if looks_like_clarification(text):
        return "clarify"
    if looks_like_refusal(text):
        return "refuse"
    return "answer"


def generate_answer(client, model: str, query: str, chunks: Sequence[dict],
                    temperature: float = 0.0, max_tokens: int = 700) -> Answer:
    """
    One call, temperature 0.

    Temperature 0 is not about answer quality -- it is about the evaluation
    being repeatable. A comparison between six runs is worthless if rerunning
    the same run moves the numbers, so every source of nondeterminism we can
    remove, we remove.
    """
    if not chunks:
        return Answer(
            text=("The retrieved filings do not contain information on this. "
                  "No relevant passages were found in the indexed 10-K filings."),
            mode="refuse")

    prompt = USER_TEMPLATE.format(passages=format_passages(chunks), query=query)
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}])
    text = (r.choices[0].message.content or "").strip()

    # Strip reasoning-model scratchpads before anything else reads the text --
    # a <think> block full of hedging would otherwise trip the refusal
    # classifier and mislabel a perfectly good answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    idxs = sorted({int(m) for m in _CITE_RE.findall(text)
                   if 1 <= int(m) <= len(chunks)})
    used = [chunks[i - 1] for i in idxs]
    cites = [_citation(c) for c in used]

    usage = {}
    if getattr(r, "usage", None):
        usage = {"prompt_tokens": r.usage.prompt_tokens,
                 "completion_tokens": r.usage.completion_tokens}

    return Answer(text=text, citations=cites, cited_idx=idxs,
                  used_chunks=used, mode=classify_mode(text), usage=usage)


def _citation(c: dict) -> str:
    who = c.get("company") or c.get("doc_title") or "Source"
    if c.get("filing_date"):
        who = f"{who} {c['filing_date']}"
    sec = c.get("section", "")
    return f"[{who}, {sec}]" if sec else f"[{who}]"


def with_sources(ans: Answer) -> str:
    """Answer text plus a resolved source list, for the UI and the demo."""
    if not ans.citations:
        return ans.text
    seen, lines = set(), []
    for c in ans.citations:
        if c not in seen:
            seen.add(c)
            lines.append(f"- {c}")
    return ans.text + "\n\n**Sources**\n" + "\n".join(lines)


# ------------------------------------------------------------- faithfulness

JUDGE_SYSTEM = """You are grading whether an answer is supported by the passages it was given.

You are NOT grading whether the answer is true in the real world, and you are
NOT grading whether it is well written. You are grading one thing: is every
factual claim in the answer traceable to the passages?

Work claim by claim. A claim is supported if a passage states it or it follows
by direct arithmetic from figures in a passage. A claim is unsupported if it
requires any outside knowledge, however obviously correct.

A refusal or a request for clarification contains no factual claims about the
filings and is therefore fully supported — score it 1.0.

Reply with ONLY a JSON object:
{"score": <0.0-1.0>, "n_claims": <int>, "unsupported": ["<claim>", ...]}
where score is (supported claims / total claims)."""


def judge_faithfulness(client, model: str, query: str, answer: str,
                       chunks: Sequence[dict]) -> Dict:
    """
    LLM-as-judge, scoped narrowly to groundedness.

    The judge is asked one mechanical question -- is each claim traceable to a
    passage -- rather than "is this a good answer". Narrow rubrics are the
    difference between a judge that agrees with itself and one that produces
    a number nobody can defend. It also returns the specific unsupported
    claims, which is what makes a failed case debuggable instead of just red.
    """
    import json
    prompt = (f"PASSAGES\n{format_passages(chunks, max_chars=1200)}\n\n"
              f"QUESTION\n{query}\n\nANSWER\n{answer}")
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=500,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": prompt}])
        raw = re.sub(r"<think>.*?</think>", "",
                     r.choices[0].message.content or "", flags=re.DOTALL)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        d = json.loads(m.group(0)) if m else {}
        return {"score": float(d.get("score", float("nan"))),
                "n_claims": int(d.get("n_claims", 0)),
                "unsupported": d.get("unsupported", [])}
    except Exception as e:
        # A judge failure must not silently become a score of zero -- that
        # would look like a pipeline regression in the report.
        return {"score": float("nan"), "n_claims": 0,
                "unsupported": [], "error": f"{type(e).__name__}: {e}"}
