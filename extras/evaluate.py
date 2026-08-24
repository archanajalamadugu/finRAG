"""
The eval harness: turn a pipeline configuration into a row of the matrix.

One function, `run_eval`, takes a retriever, a reranker and the question set
and produces a `RunResult`. Every row of the experiment matrix is the same
call with different arguments, which is the property that makes the two
required comparisons trustworthy: A3 and B2 are not two pipelines that happen
to be similar, they are one pipeline with one flag changed.

Two things this harness does that a simpler one would not:

  * It times each stage separately. The reranking impact analysis has to state
    its cost as well as its benefit, and "the whole thing took 6 seconds" does
    not tell you whether the reranker or the generator is responsible.

  * It records the pre-rerank top-k alongside the post-rerank top-k for every
    question, so the reranking analysis can say *what moved*, not merely that
    the average went up. Aggregate metrics hide the case where a reranker
    fixes three questions and breaks two.

Labelling
---------
`relevant_chunks` in the eval file is a list of chunk ids. Chunk ids are
strategy-specific (`fixed_split-00123` vs `semantic-00087`), so ground truth
cannot be a raw id list shared across runs. `label_by_predicate` resolves
labels per chunk-set from a stable description -- company + section +
required substring -- which is the only form of ground truth that survives
re-chunking. Re-labelling by hand for each of six runs would guarantee
inconsistency; this makes the label a property of the corpus, not the index.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .metrics import (RunResult, behaviour_verdict, mrr_at_k, ndcg_at_k,
                      numeric_verdict, recall_at_k)
from .rerank import IdentityReranker, Timer, rank_movement
from .retrieve import Hit, source_diversity


# ------------------------------------------------------------------ labelling

def label_by_predicate(chunks: Sequence[dict], spec: Dict) -> List[str]:
    """
    Resolve a portable ground-truth spec into chunk ids for THIS chunk set.

    A spec looks like:
        {"company": "NVIDIA", "section_contains": "Item 8",
         "text_contains_any": ["Total revenue", "Revenue"]}

    Every field is optional and all supplied fields must match. Matching is
    case-insensitive because filings are inconsistent about capitalising
    section headings.
    """
    out = []
    comp = (spec.get("company") or "").lower()
    sec = (spec.get("section_contains") or "").lower()
    any_of = [s.lower() for s in spec.get("text_contains_any", [])]
    all_of = [s.lower() for s in spec.get("text_contains_all", [])]
    kind = spec.get("kind")

    for c in chunks:
        if comp and comp != (c.get("company", "") or "").lower():
            continue
        if sec and sec not in (c.get("section", "") or "").lower():
            continue
        if kind and kind != c.get("kind"):
            continue
        t = (c.get("text", "") or "").lower()
        if any_of and not any(s in t for s in any_of):
            continue
        if all_of and not all(s in t for s in all_of):
            continue
        out.append(c["chunk_id"])
    return out


def resolve_labels(questions: Sequence[Dict], chunks: Sequence[dict]) -> Dict[str, List[str]]:
    """Ground truth for every question, resolved against one chunk set."""
    labels: Dict[str, List[str]] = {}
    for q in questions:
        specs = q.get("relevant_specs") or []
        ids: List[str] = []
        for s in specs:
            ids.extend(label_by_predicate(chunks, s))
        labels[q["id"]] = sorted(set(ids))
    return labels


def resolve_labels_multi(questions: Sequence[Dict],
                         chunk_sets: Dict[str, Sequence[dict]]) -> Dict:
    """
    Resolve ground truth against EVERY chunk set at once, and work out which
    questions are legitimately scorable.

    This exists to close a hole that would otherwise flatter the naive
    baseline. Ground truth is a predicate over chunk content, so a strategy
    that destroys the evidence -- fixed-size windows cutting a segment table
    so no single chunk holds the header and its figures -- resolves to *zero*
    labelled chunks. Zero labels means recall is NaN, NaN is excluded from the
    mean, and the strategy silently skips the exact question it just failed.
    The baseline would look better precisely because it was worse.

    So: a question is `scorable` if at least one chunk set can label it. For
    any scorable question, a chunk set with no labels scores 0.0, not NaN.
    NaN is reserved for questions no strategy could label -- which is a broken
    spec, not a result, and is reported separately.
    """
    per_set = {name: resolve_labels(questions, cs) for name, cs in chunk_sets.items()}
    scorable, unlabelled = set(), set()
    for q in questions:
        if q.get("expect_refusal") or q.get("expect_clarification"):
            continue
        if any(per_set[n].get(q["id"]) for n in per_set):
            scorable.add(q["id"])
        else:
            unlabelled.add(q["id"])
    return {"per_set": per_set, "scorable": scorable, "unlabelled": unlabelled}


def evidence_survival(multi: Dict) -> str:
    """
    Which chunk sets lost the evidence, per question.

    This table is a finding in its own right and belongs in the chunking
    comparison: it shows *mechanically* what a boundary policy did to the
    corpus, before any retrieval or generation is involved. A zero in the
    fixed-split column against a table_lookup row is the table-splitting
    failure, made visible.
    """
    names = list(multi["per_set"])
    qids = sorted({q for n in names for q in multi["per_set"][n]})
    w = max(len(n) for n in names) + 2
    lines = ["question  " + "".join(f"{n:>{w}}" for n in names)]
    for qid in qids:
        if qid not in multi["scorable"] and qid not in multi["unlabelled"]:
            continue                                  # behaviour case
        cells = "".join(f"{len(multi['per_set'][n].get(qid, [])):>{w}d}" for n in names)
        flag = "" if qid in multi["scorable"] else "   <-- no strategy labels this"
        lines.append(f"  {qid:<8s}{cells}{flag}")
    return "\n".join(lines)


def label_coverage(questions: Sequence[Dict], labels: Dict[str, List[str]]) -> str:
    """
    Report which questions actually got labelled, before any run happens.

    Silent label failure is the most expensive bug in an eval harness: a spec
    that matches nothing produces recall of NaN, which is excluded from the
    mean, and the run looks fine while measuring less than you think. This is
    printed before the matrix runs so a broken spec is caught while it is
    still cheap to fix.
    """
    lines = ["question   type              n_labelled"]
    unlabelled = []
    for q in questions:
        if q.get("expect_refusal") or q.get("expect_clarification"):
            lines.append(f"  {q['id']}     {q['type']:<16s}  behaviour case")
            continue
        n = len(labels.get(q["id"], []))
        flag = "" if n else "   <-- NO LABELS"
        if not n:
            unlabelled.append(q["id"])
        lines.append(f"  {q['id']}     {q['type']:<16s}  {n:>3d}{flag}")
    if unlabelled:
        lines.append("")
        lines.append(f"WARNING: {len(unlabelled)} scored questions have no labels "
                     f"({', '.join(unlabelled)}).")
        lines.append("Their retrieval metrics will be excluded, not zero. "
                     "Fix the specs before trusting the matrix.")
    return "\n".join(lines)


# ------------------------------------------------------------------ the run

def run_eval(run_id: str,
             retriever,
             questions: Sequence[Dict],
             *,
             mode: str = "hybrid",
             reranker=None,
             client=None,
             chat_model: str = "",
             top_k: int = 5,
             per_retriever: int = 20,
             judge: bool = True,
             judge_model: Optional[str] = None,
             generate: bool = True,
             scorable_ids: Optional[Sequence[str]] = None,
             config: Optional[Dict] = None,
             progress: bool = True) -> RunResult:
    """
    Execute one row of the experiment matrix.

    `generate=False` runs retrieval only. Worth using for the first pass over
    all six runs: retrieval metrics are what the two required comparisons turn
    on, they cost nothing but embeddings, and getting them for the whole matrix
    before spending any generation tokens means a broken config surfaces in
    seconds rather than after twenty LLM calls.
    """
    from extras.generate import generate_answer, judge_faithfulness

    reranker = reranker or IdentityReranker()
    labels = resolve_labels(questions, retriever.chunks)
    # Questions another chunk set could label. See `resolve_labels_multi`:
    # having no labels here is a result (the evidence did not survive
    # chunking), not a reason to skip the question.
    scorable = set(scorable_ids) if scorable_ids is not None else None
    per_q: List[Dict] = []

    for n, q in enumerate(questions, 1):
        if progress:
            print(f"\r  {run_id}: {n}/{len(questions)}  {q['id']}   ", end="")
        t = Timer()
        rec: Dict = {"id": q["id"], "type": q["type"], "question": q["q"],
                     "expect_refusal": bool(q.get("expect_refusal")),
                     "expect_clarification": bool(q.get("expect_clarification"))}

        # --- retrieve (wide) ------------------------------------------------
        wide_k = per_retriever if reranker.name != "none" else top_k
        hits: List[Hit] = t.time(
            "retrieve", retriever.retrieve, q["q"], mode=mode,
            top_k=wide_k, per_retriever=per_retriever)

        # --- rerank ---------------------------------------------------------
        before = list(hits[:top_k])
        final: List[Hit] = t.time(
            "rerank", reranker.rerank, q["q"], retriever.chunks, hits, top_k)
        rec["rank_movement"] = rank_movement(before, final)

        top_chunks = [retriever.chunks[h.idx] for h in final]
        rec["retrieved"] = [c["chunk_id"] for c in top_chunks]
        rec["retrieved_wide"] = [retriever.chunks[h.idx]["chunk_id"] for h in hits[:10]]
        rec["retrieved_meta"] = [
            {"company": c.get("company"), "section": c.get("section"),
             "kind": c.get("kind")} for c in top_chunks]

        # --- retrieval metrics ----------------------------------------------
        rel = labels.get(q["id"], [])
        rec["n_labels"] = len(rel)
        # No labels in THIS chunk set, but another set could label the same
        # question: the evidence did not survive this chunking policy. That is
        # a zero, not an abstention. Scoring it NaN would let the strategy skip
        # the very question its boundaries just broke.
        rec["evidence_lost"] = bool(
            not rel and scorable is not None and q["id"] in scorable)
        if rec["evidence_lost"]:
            rec["recall"] = rec["mrr"] = rec["ndcg"] = 0.0
        else:
            rec["recall"] = recall_at_k(rec["retrieved"], rel, k=top_k)
            # MRR and nDCG read the wide list: a relevant chunk at rank 8 is a
            # different situation from one that is absent, and top-5-only
            # metrics cannot tell those apart when diagnosing a reranker.
            rec["mrr"] = mrr_at_k(rec["retrieved_wide"], rel, k=10)
            rec["ndcg"] = ndcg_at_k(rec["retrieved_wide"], rel, k=10)

        need = q.get("min_sources")
        if need:
            got = source_diversity(top_chunks)
            rec["source_diversity"] = got
            rec["diversity_ok"] = 1.0 if got >= need else 0.0

        # --- generate --------------------------------------------------------
        if generate and client is not None:
            ans = t.time("generate", generate_answer, client, chat_model,
                         q["q"], top_chunks)
            rec["answer"] = ans.text
            rec["answer_mode"] = ans.mode
            rec["citations"] = ans.citations
            rec["n_cited"] = len(ans.cited_idx)

            bv = behaviour_verdict(ans.text,
                                   expect_refusal=rec["expect_refusal"],
                                   expect_clarification=rec["expect_clarification"])
            rec["got"] = bv["got"]
            rec["behaviour_correct"] = bv["correct"]

            if q.get("numeric") and q.get("expected_value") is not None:
                rec["numeric_verdict"] = numeric_verdict(
                    ans.text, float(q["expected_value"]))

            if judge and not (rec["expect_refusal"] or rec["expect_clarification"]):
                j = t.time("judge", judge_faithfulness, client,
                           judge_model or chat_model, q["q"], ans.text, top_chunks)
                rec["faithfulness"] = j["score"]
                rec["unsupported_claims"] = j.get("unsupported", [])

        rec["latency"] = t.as_dict()
        # Judge time is measurement, not product. Excluding it keeps the
        # latency column answering "how long does the analyst wait".
        rec["latency_total"] = rec["latency"]["total"] - rec["latency"].get("judge", 0.0)
        per_q.append(rec)

    if progress:
        print(f"\r  {run_id}: done ({len(questions)} questions)          ")

    cfg = dict(config or {})
    cfg.update({"mode": mode, "reranker": reranker.name, "top_k": top_k,
                "per_retriever": per_retriever, "generated": bool(generate and client)})
    return RunResult(run_id=run_id, config=cfg, per_question=per_q)


# ------------------------------------------------------------------ failures

def failure_report(run: RunResult, k: int = 5, limit: int = 8) -> str:
    """
    The cases that went wrong, with enough context to act on them.

    This is the input to the "iterations you tried" section of the write-up.
    An average tells you whether to keep going; a failure list tells you what
    to change, which is the difference between a report and a project.
    """
    lines = [f"FAILURES · {run.run_id}", "=" * 60]
    shown = 0
    for r in run.per_question:
        lost = bool(r.get("evidence_lost"))
        bad_retrieval = (r.get("recall") == 0.0 and r.get("n_labels", 0) > 0) or lost
        bad_behaviour = r.get("behaviour_correct") is False
        bad_numeric = r.get("numeric_verdict") in ("wrong", "no_number", "scale_slip")
        unfaithful = (r.get("faithfulness") or 1.0) < 0.9
        bad_div = r.get("diversity_ok") == 0.0
        if not (bad_retrieval or bad_behaviour or bad_numeric or unfaithful or bad_div):
            continue
        if shown >= limit:
            lines.append(f"... and more; {limit} shown")
            break
        shown += 1
        why = [n for n, c in (("EVIDENCE LOST IN CHUNKING", lost),
                              ("retrieval", bad_retrieval and not lost),
                              ("behaviour", bad_behaviour),
                              ("numeric", bad_numeric), ("faithfulness", unfaithful),
                              ("diversity", bad_div)) if c]
        lines.append(f"\n{r['id']} [{r['type']}] — {', '.join(why)}")
        lines.append(f"  Q: {r['question']}")
        if lost:
            lines.append("  No chunk in this set contains the labelled evidence — "
                         "this strategy's boundaries destroyed it.")
        if r.get("n_labels"):
            lines.append(f"  recall@{k}={r.get('recall')}  mrr={r.get('mrr'):.3f}  "
                         f"labels={r['n_labels']}")
        if r.get("numeric_verdict"):
            lines.append(f"  numeric: {r['numeric_verdict']}")
        if r.get("diversity_ok") == 0.0:
            lines.append(f"  sources: {r.get('source_diversity')} distinct companies retrieved")
        meta = r.get("retrieved_meta") or []
        if meta:
            lines.append("  got: " + "; ".join(
                f"{m.get('company')}/{m.get('section','')[:28]}" for m in meta[:5]))
        if r.get("unsupported_claims"):
            lines.append(f"  unsupported: {r['unsupported_claims'][:2]}")
        if r.get("answer"):
            lines.append(f"  A: {r['answer'][:200].replace(chr(10),' ')}")
    if shown == 0:
        lines.append("\nNo failures. Check that labels resolved — an unlabelled "
                     "eval set also produces no failures.")
    return "\n".join(lines)


def save_run(run: RunResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"run_id": run.run_id, "config": run.config,
         "summary": run.summary(), "per_question": run.per_question},
        indent=2, default=str))
    return path


def load_run(path: Path) -> RunResult:
    d = json.loads(Path(path).read_text())
    return RunResult(run_id=d["run_id"], config=d.get("config", {}),
                     per_question=d.get("per_question", []))
