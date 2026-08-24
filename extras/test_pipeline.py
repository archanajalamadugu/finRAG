"""
Offline tests for the retrieval, reranking, generation and metric layers.

No network and no API key. The embedder is the TF-IDF/SVD stand-in from
`tests.mock_embed` and the LLM is a scripted fake, so every one of these runs
in Colab-free CI and, more importantly, catches the class of bug that is
expensive to find at 11pm on a Sunday: a metric that silently returns zero, a
fusion that drops a retriever, a refusal classifier that fires on a normal
answer.

Run:  python3 -m tests.test_pipeline
"""
from __future__ import annotations

import sys
import numpy as np

from extras.retrieve import (BM25Index, Hit, HybridRetriever, NumpyDenseIndex,
                          reciprocal_rank_fusion, source_diversity, tokenize)
from extras.rerank import IdentityReranker, LLMListwiseReranker, Timer, rank_movement
from extras.metrics import (RunResult, behaviour_verdict, comparison_table,
                         delta_table, extract_numbers, looks_like_clarification,
                         looks_like_refusal, mrr_at_k, ndcg_at_k, numeric_match,
                         numeric_verdict, percentile, precision_at_k,
                         recall_at_k, refusal_precision)
from extras.evaluate import (evidence_survival, label_by_predicate, label_coverage,
                          resolve_labels, resolve_labels_multi)
from extras.generate import Answer, classify_mode, format_passages, with_sources
from tests.mock_embed import MockEmbedder

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {extra}")


def section(t):
    print(f"\n{t}")


# ------------------------------------------------------------------ fixtures

CHUNKS = [
    {"chunk_id": "c0", "company": "NVIDIA", "filing_date": "2025-02-26",
     "section": "Item 8 - Financial Statements", "kind": "table",
     "text": "Revenue by segment | FY2025 | FY2024\nData Center | $ | 115,186 | $ | 47,525\nGaming | $ | 11,350 | $ | 10,447"},
    {"chunk_id": "c1", "company": "NVIDIA", "filing_date": "2025-02-26",
     "section": "Item 7 - Management's Discussion and Analysis", "kind": "text",
     "text": "Revenue increased driven by strong demand for our accelerated computing platforms in the data center market."},
    {"chunk_id": "c2", "company": "AMD", "filing_date": "2025-02-05",
     "section": "Item 8 - Financial Statements", "kind": "table",
     "text": "Segment revenue | 2024 | 2023\nData Center | $ | 12,580 | $ | 6,496\nClient | $ | 7,052 | $ | 4,651"},
    {"chunk_id": "c3", "company": "Intel", "filing_date": "2025-01-31",
     "section": "Item 1A - Risk Factors", "kind": "text",
     "text": "We depend on a limited number of customers for a substantial portion of our revenue, and export controls may restrict sales."},
    {"chunk_id": "c4", "company": "Broadcom", "filing_date": "2024-12-13",
     "section": "Item 7 - Management's Discussion and Analysis", "kind": "text",
     "text": "Gross margin improved due to a favourable product mix weighted toward infrastructure software."},
]


class FakeChat:
    """Scripted OpenAI-shaped client. Returns whatever it is told to."""

    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        self.calls.append(kw)
        text = self.reply(kw) if callable(self.reply) else self.reply

        class M:
            content = text

        class C:
            message = M()

        class R:
            choices = [C()]
            usage = None
        return R()


# ------------------------------------------------------------------ tokenise

section("tokenisation")
check("figures survive as one token", "47,525" in tokenize("Data Center 47,525"))
check("percentages survive", "12.4%" in tokenize("margin was 12.4%"))
check("decimals not split", "1.5" in tokenize("about 1.5 billion"))
check("stopwords dropped", "the" not in tokenize("the revenue"))
check("case folded", tokenize("Data Center") == ["data", "center"])
check("empty text safe", tokenize("") == [])
check("None-ish safe", tokenize(None) == [])


# --------------------------------------------------------------- dense index

section("dense index")
emb = MockEmbedder()
emb.fit([c["text"] for c in CHUNKS])
vecs = emb([c["text"] for c in CHUNKS])
di = NumpyDenseIndex().add(vecs)
check("index length", len(di) == 5)
q = emb(["data center revenue"])[0]
hits = di.search(q, k=3)
check("returns k hits", len(hits) == 3)
check("ranks are 1-based and ordered", [h.rank for h in hits] == [1, 2, 3])
check("scores descend", all(hits[i].score >= hits[i + 1].score for i in range(2)))
check("k larger than corpus is clamped", len(di.search(q, k=99)) == 5)
check("empty index returns nothing", NumpyDenseIndex().search(q, k=5) == [])
try:
    NumpyDenseIndex().add(np.zeros(5))
    check("1-D vectors rejected", False)
except ValueError:
    check("1-D vectors rejected", True)


# -------------------------------------------------------------- sparse index

section("bm25")
bm = BM25Index().add([c["text"] for c in CHUNKS])
check("index length", len(bm) == 5)
h = bm.search("Data Center", k=3)
check("exact segment name retrieves", len(h) > 0)
check("both data-center tables found", {hh.idx for hh in h} >= {0, 2})
check("figure string retrieves the right table",
      bm.search("115,186", k=1)[0].idx == 0)
check("unmatched query returns empty", bm.search("zzzzqqq", k=5) == [])
check("idf never negative", all(v >= 0 for v in bm.idf.values()))
check("empty corpus safe", BM25Index().add([]).search("x", k=3) == [])
check("sparse beats dense on a literal",
      bm.search("115,186", k=1)[0].idx == 0)


# ----------------------------------------------------------------------- rrf

section("reciprocal rank fusion")
r1 = [Hit(idx=0, score=.9, source="dense", rank=1),
      Hit(idx=1, score=.8, source="dense", rank=2)]
r2 = [Hit(idx=1, score=5.0, source="sparse", rank=1),
      Hit(idx=2, score=4.0, source="sparse", rank=2)]
f = reciprocal_rank_fusion({"dense": r1, "sparse": r2}, top_k=5)
check("union of both lists", {h.idx for h in f} == {0, 1, 2})
check("doc in both lists wins", f[0].idx == 1)
check("components recorded", f[0].components == {"dense": 2, "sparse": 1})
check("ranks reassigned", [h.rank for h in f] == [1, 2, 3])
check("top_k respected", len(reciprocal_rank_fusion({"a": r1, "b": r2}, top_k=2)) == 2)
check("ignores raw score scale",
      reciprocal_rank_fusion(
          {"a": [Hit(idx=7, score=1e9, source="a", rank=3)],
           "b": [Hit(idx=8, score=1e-9, source="b", rank=1)]}, top_k=2)[0].idx == 8)
check("empty input safe", reciprocal_rank_fusion({}, top_k=5) == [])


# ------------------------------------------------------------------ retriever

section("hybrid retriever")
ret = HybridRetriever.build(CHUNKS, emb, vectors=vecs)
check("chunks held", len(ret.chunks) == 5)
check("dense mode works", len(ret.retrieve("data center", mode="dense", top_k=3)) == 3)
check("sparse mode works", len(ret.retrieve("Data Center", mode="sparse", top_k=3)) == 3)
hy = ret.retrieve("Data Center revenue", mode="hybrid", top_k=3)
check("hybrid mode works", len(hy) == 3)
check("hybrid hits are rrf", all(h.source == "rrf" for h in hy))
try:
    ret.retrieve("x", mode="nonsense")
    check("bad mode rejected", False)
except ValueError:
    check("bad mode rejected", True)
try:
    HybridRetriever.build(CHUNKS, emb, vectors=vecs[:3])
    check("vector/chunk mismatch caught", False)
except ValueError:
    check("vector/chunk mismatch caught", True)
check("explain renders", "QUERY" in ret.explain("data center", top_k=2))
check("query embedding cached",
      (ret.retrieve("same q", top_k=1), "same q" in ret._qcache)[1])
check("source diversity counts companies",
      source_diversity(CHUNKS) == 4)
check("diversity ignores blanks",
      source_diversity([{"company": ""}, {"company": "AMD"}]) == 1)


# ------------------------------------------------------------------ rerankers

section("reranking")
idr = IdentityReranker()
wide = ret.retrieve("data center", mode="hybrid", top_k=5)
same = idr.rerank("data center", CHUNKS, wide, top_k=3)
check("identity preserves order", [h.idx for h in same] == [h.idx for h in wide[:3]])
check("identity renumbers ranks", [h.rank for h in same] == [1, 2, 3])
check("identity is named none", idr.name == "none")

llm = LLMListwiseReranker(FakeChat("2, 0, 1"), "fake-model")
out = llm.rerank("q", CHUNKS, wide[:3], top_k=3)
check("llm order applied", [wide[i].idx for i in (2, 0, 1)] == [h.idx for h in out])
check("pre-rerank rank recorded", "pre_rerank" in out[0].components)

out = LLMListwiseReranker(FakeChat("garbage no digits"), "m").rerank("q", CHUNKS, wide[:3], 3)
check("unparseable reply falls back to retrieval order",
      [h.idx for h in out] == [h.idx for h in wide[:3]])
out = LLMListwiseReranker(FakeChat("99, 1, 1, -4"), "m").rerank("q", CHUNKS, wide[:3], 3)
check("out-of-range and duplicate indices dropped",
      sorted(h.idx for h in out) == sorted(h.idx for h in wide[:3]))
check("empty candidate list safe", llm.rerank("q", CHUNKS, [], 5) == [])


class Boom:
    chat = property(lambda s: s)
    completions = property(lambda s: s)

    def create(self, **kw):
        raise RuntimeError("nebius down")


out = LLMListwiseReranker(Boom(), "m").rerank("q", CHUNKS, wide[:3], 3)
check("api failure degrades to retrieval order",
      [h.idx for h in out] == [h.idx for h in wide[:3]])

mv = rank_movement(wide[:3], out)
check("movement reports unchanged", mv["unchanged"] is True)
mv2 = rank_movement(wide[:3], llm.rerank("q", CHUNKS, wide[:3], 3))
check("movement detects top-1 change", mv2["top1_changed"] is True)


# --------------------------------------------------------------------- timer

section("timing")
t = Timer()
t.time("a", lambda: sum(range(1000)))
t.time("b", lambda: sum(range(1000)))
t.time("a", lambda: sum(range(1000)))
d = t.as_dict()
check("stages accumulate by name", set(d) == {"a", "b", "total"})
check("total is the sum", abs(d["total"] - (d["a"] + d["b"])) < 1e-9)
check("return value passed through", t.time("c", lambda x: x * 2, 21) == 42)


# ------------------------------------------------------------- retrieval math

section("retrieval metrics")
check("recall counts labelled hits", recall_at_k(["a", "b", "c"], ["a", "d"], 3) == 0.5)
check("recall respects k", recall_at_k(["x", "x", "a"], ["a"], 2) == 0.0)
check("recall unlabelled is NaN", recall_at_k(["a"], [], 5) != recall_at_k(["a"], [], 5))
check("precision", precision_at_k(["a", "b"], ["a"], 2) == 0.5)
check("mrr rank 1", mrr_at_k(["a", "b"], ["a"], 10) == 1.0)
check("mrr rank 3", abs(mrr_at_k(["x", "y", "a"], ["a"], 10) - 1 / 3) < 1e-9)
check("mrr miss is zero", mrr_at_k(["x"], ["a"], 10) == 0.0)
check("ndcg perfect is 1", abs(ndcg_at_k(["a", "b"], ["a", "b"], 10) - 1.0) < 1e-9)
check("ndcg rewards higher placement",
      ndcg_at_k(["a", "x"], ["a"], 10) > ndcg_at_k(["x", "a"], ["a"], 10))
check("ndcg miss is zero", ndcg_at_k(["x", "y"], ["a"], 10) == 0.0)
check("ndcg with grades", 0 < ndcg_at_k(["b", "a"], ["a", "b"], 10,
                                        grades={"a": 2.0, "b": 1.0}) < 1.0)
check("percentile p50", percentile([1, 2, 3], 50) == 2)
check("percentile interpolates", abs(percentile([0, 10], 95) - 9.5) < 1e-9)


# ------------------------------------------------------------- number parsing

section("numeric scoring")
check("plain integer", extract_numbers("Revenue was 60922")[0] == 60922)
check("comma grouping", extract_numbers("$60,922")[0] == 60922)
check("million scaled", extract_numbers("$60,922 million")[0] == 60922e6)
check("billion scaled", extract_numbers("60.9 billion")[0] == 60.9e9)
check("percent kept raw", extract_numbers("margin of 12.4%")[0] == 12.4)
check("parentheses negate", extract_numbers("(1,234)")[0] == -1234)
check("no numbers", extract_numbers("no figures here") == [])
check("million vs billion match", numeric_match("$60.9 billion", 60922e6, rel_tol=0.01))
check("1% tolerance holds", numeric_match("60,500", 60922, rel_tol=0.01))
check("2% is outside tolerance", not numeric_match("59,000", 60922, rel_tol=0.01))
check("exact verdict", numeric_verdict("Revenue was 60,922", 60922) == "exact")
check("scale slip caught separately",
      numeric_verdict("Revenue was 60,922", 60922e6) == "scale_slip")
check("wrong verdict", numeric_verdict("Revenue was 12", 60922) == "wrong")
check("no_number verdict", numeric_verdict("It grew a lot", 60922) == "no_number")


# ----------------------------------------------------------------- behaviour

section("behaviour scoring")
check("refusal detected",
      looks_like_refusal("The filings do not contain forward guidance."))
check("normal answer is not a refusal",
      not looks_like_refusal("Revenue was $60,922 million [1]."))
check("clarification detected",
      looks_like_clarification("Which company did you mean?"))
check("statement without ? is not a clarification",
      not looks_like_clarification("Clarify the company."))
bv = behaviour_verdict("This is not in the 10-K.", expect_refusal=True)
check("correct refusal scores true", bv["correct"] is True)
bv = behaviour_verdict("Revenue was $1bn.", expect_refusal=True)
check("guessing instead of refusing scores false", bv["correct"] is False)
bv = behaviour_verdict("Which company do you mean?", expect_clarification=True)
check("correct clarification scores true", bv["correct"] is True)
bv = behaviour_verdict("Revenue was $1bn [1].")
check("normal answer to normal question scores true", bv["correct"] is True)
bv = behaviour_verdict("Not disclosed in the filings.")
check("over-refusal on a normal question scores false", bv["correct"] is False)

rp = refusal_precision([
    {"expect_refusal": True, "got": "refuse"},
    {"expect_refusal": False, "expect_clarification": False, "got": "refuse"},
    {"expect_refusal": True, "got": "answered"},
    {"expect_refusal": False, "got": "answered"},
])
check("over-refusals counted", rp["over_refusals"] == 1)
check("missed refusals counted", rp["missed_refusals"] == 1)
check("refusal precision", rp["refusal_precision"] == 0.5)


# ------------------------------------------------------------------ labelling

section("ground-truth labelling")
spec = {"company": "NVIDIA", "section_contains": "Item 8",
        "text_contains_any": ["Data Center"]}
check("predicate resolves to the right chunk",
      label_by_predicate(CHUNKS, spec) == ["c0"])
check("company filter applied",
      label_by_predicate(CHUNKS, {"company": "AMD"}) == ["c2"])
check("kind filter applied",
      set(label_by_predicate(CHUNKS, {"kind": "table"})) == {"c0", "c2"})
check("contains_all requires every term",
      label_by_predicate(CHUNKS, {"text_contains_all": ["Data Center", "Gaming"]}) == ["c0"])
check("no match returns empty",
      label_by_predicate(CHUNKS, {"company": "Qualcomm"}) == [])
check("matching is case-insensitive",
      label_by_predicate(CHUNKS, {"company": "nvidia", "kind": "table"}) == ["c0"])

qs = [{"id": "Q1", "type": "table_lookup", "q": "?", "relevant_specs": [spec]},
      {"id": "Q2", "type": "table_lookup", "q": "?",
       "relevant_specs": [{"company": "Nobody"}]},
      {"id": "Q3", "type": "unanswerable", "q": "?", "expect_refusal": True}]
labels = resolve_labels(qs, CHUNKS)
check("labels resolved per question", labels["Q1"] == ["c0"])
check("unmatched spec yields empty list", labels["Q2"] == [])
cov = label_coverage(qs, labels)
check("coverage flags unlabelled questions", "NO LABELS" in cov and "Q2" in cov)
check("coverage skips behaviour cases", "behaviour case" in cov)


section("evidence survival across chunk sets")
# "rich" keeps the table whole; "poor" has split it, so no single chunk holds
# both the segment name and its figures. This is the fixed-vs-semantic failure
# in miniature.
rich = [{"chunk_id": "r0", "company": "NVIDIA", "kind": "table",
         "section": "Item 8", "text": "Data Center | 115,186 | 47,525"}]
poor = [{"chunk_id": "p0", "company": "NVIDIA", "kind": "text",
         "section": "Item 8", "text": "Revenue by segment"},
        {"chunk_id": "p1", "company": "NVIDIA", "kind": "text",
         "section": "Item 8", "text": "115,186 | 47,525"}]
mq = [{"id": "T1", "type": "table_lookup", "q": "?",
       "relevant_specs": [{"text_contains_all": ["Data Center", "47,525"]}]},
      {"id": "T2", "type": "table_lookup", "q": "?",
       "relevant_specs": [{"company": "Nobody"}]},
      {"id": "T3", "type": "unanswerable", "q": "?", "expect_refusal": True}]
multi = resolve_labels_multi(mq, {"rich": rich, "poor": poor})
check("question labelled by one set is scorable", "T1" in multi["scorable"])
check("question no set can label is flagged unlabelled", "T2" in multi["unlabelled"])
check("behaviour case excluded from both",
      "T3" not in multi["scorable"] and "T3" not in multi["unlabelled"])
check("rich set resolves the label", multi["per_set"]["rich"]["T1"] == ["r0"])
check("poor set resolves nothing", multi["per_set"]["poor"]["T1"] == [])
surv = evidence_survival(multi)
check("survival table names both sets", "rich" in surv and "poor" in surv)
check("survival table flags the unlabellable question", "no strategy labels" in surv)

# The fix itself: the strategy that destroyed the evidence must score 0, not NaN.
from extras.evaluate import run_eval as _run_eval
_poor_ret = HybridRetriever.build(poor, MockEmbedder().fit([c["text"] for c in poor]))
naive = _run_eval("poor_naive", _poor_ret, mq[:1], mode="dense", top_k=2,
                  generate=False, progress=False)
scored_ = _run_eval("poor_scored", _poor_ret, mq[:1], mode="dense", top_k=2,
                    generate=False, scorable_ids=multi["scorable"], progress=False)
check("without the fix an unlabelled question abstains (NaN)",
      naive.per_question[0]["recall"] != naive.per_question[0]["recall"])
check("with the fix destroyed evidence scores zero",
      scored_.per_question[0]["recall"] == 0.0)
check("evidence_lost flagged on the record",
      scored_.per_question[0]["evidence_lost"] is True)
check("evidence_lost counted in the summary",
      scored_.summary()["evidence_lost"] == 1)
check("summary recall is 0.0, not NaN", scored_.summary()["recall@5"] == 0.0)
from extras.evaluate import failure_report as _fr
check("failure report names the cause",
      "EVIDENCE LOST IN CHUNKING" in _fr(scored_, limit=3))


# ------------------------------------------------------------------ prompting

section("prompt assembly")
p = format_passages(CHUNKS[:2])
check("passages numbered from 1", p.startswith("[1]"))
check("company in the header", "NVIDIA" in p)
check("section in the header", "Item 8" in p)
check("filing date in the header", "2025-02-26" in p)
check("truncation marked",
      "[passage truncated]" in format_passages(
          [{"text": "x" * 5000, "company": "A"}], max_chars=100))
check("mode classify: answer", classify_mode("Revenue was $1m [1].") == "answer")
check("mode classify: refuse", classify_mode("That is not in the filings.") == "refuse")
check("mode classify: clarify", classify_mode("Which company do you mean?") == "clarify")
a = Answer(text="x", citations=["[NVIDIA 2025, Item 8]", "[NVIDIA 2025, Item 8]"])
check("sources deduplicated", with_sources(a).count("NVIDIA") == 1)
check("no citations renders plain", with_sources(Answer(text="x")) == "x")


# ----------------------------------------------------------------- reporting

section("run reporting")
run_a = RunResult("A3", {"mode": "hybrid"}, [
    {"id": "Q1", "type": "table_lookup", "recall": 0.5, "mrr": 0.5, "ndcg": 0.5,
     "faithfulness": 1.0, "numeric_verdict": "exact", "latency_total": 2.0},
    {"id": "Q2", "type": "table_lookup", "recall": 1.0, "mrr": 1.0, "ndcg": 1.0,
     "faithfulness": 0.8, "numeric_verdict": "wrong", "latency_total": 4.0},
    {"id": "Q3", "type": "unanswerable", "expect_refusal": True,
     "behaviour_correct": True, "got": "refuse", "latency_total": 1.0},
])
s = run_a.summary()
check("recall averaged over scored questions only", s["recall@5"] == 0.75)
check("behaviour scored separately", s["behaviour_correct"] == 1.0)
check("numeric exact rate", s["numeric_exact"] == 0.5)
check("p50 latency", s["latency_p50"] == 2.0)
check("config echoed into summary", s["cfg_mode"] == "hybrid")

run_b = RunResult("B2", {"mode": "hybrid"}, [
    dict(r, recall=min(1.0, r.get("recall", 0) + 0.25)) if "recall" in r else r
    for r in run_a.per_question])
tbl = comparison_table([run_a, run_b])
check("comparison table has both runs", "A3" in tbl and "B2" in tbl)
check("comparison table is markdown", tbl.count("|") > 10)
d = delta_table(run_a, run_b)
check("delta table signs the change", "+" in d)
check("delta table names both runs", "A3" in d and "B2" in d)

nanrun = RunResult("N", {}, [{"id": "Q", "type": "t", "recall": float("nan")}])
check("all-NaN metric renders as em dash", "—" in comparison_table([nanrun]))


section("plain-English scorecard (the Week 2 bonus deliverable)")
from extras.metrics import (_evidence_cell, _grounded_cell, diagnose,
                         failure_summary, simple_scorecard)

cases = [
    # (record, expected evidence cell, expected diagnosis prefix)
    ({"id": "A", "type": "table_lookup", "question": "clean hit", "recall": 1.0,
      "faithfulness": 1.0, "n_cited": 2, "answer_mode": "answer",
      "numeric_verdict": "exact"}, "Yes", "correct"),
    ({"id": "B", "type": "table_lookup", "question": "wrong chunk", "recall": 0.0,
      "faithfulness": 1.0, "n_cited": 1, "answer_mode": "answer"},
     "No", "retrieval"),
    ({"id": "C", "type": "risk_synthesis", "question": "partial", "recall": 0.5,
      "faithfulness": 1.0, "n_cited": 2, "answer_mode": "answer"},
     "Partly", "retrieval"),
    ({"id": "D", "type": "mdna_narrative", "question": "good evidence bad answer",
      "recall": 1.0, "faithfulness": 0.5, "n_cited": 1, "answer_mode": "answer"},
     "Yes", "generation"),
    ({"id": "E", "type": "table_lookup", "question": "units dropped", "recall": 1.0,
      "faithfulness": 1.0, "n_cited": 1, "answer_mode": "answer",
      "numeric_verdict": "scale_slip"}, "Yes", "generation"),
    ({"id": "F", "type": "unanswerable", "question": "not in corpus",
      "expect_refusal": True, "behaviour_correct": True}, "No evidence exists",
     "correct restraint"),
    ({"id": "G", "type": "unanswerable", "question": "guessed anyway",
      "expect_refusal": True, "behaviour_correct": False}, "No evidence exists",
     "generation"),
    ({"id": "H", "type": "table_lookup", "question": "chunking ate it",
      "evidence_lost": True, "recall": 0.0}, "No", "retrieval"),
    ({"id": "I", "type": "mdna_narrative", "question": "no citation", "recall": 1.0,
      "faithfulness": 1.0, "n_cited": 0, "answer_mode": "answer"},
     "Yes", "generation"),
]
for rec, ev, diag in cases:
    check(f"evidence cell · {rec['id']}", _evidence_cell(rec) == ev,
          f"got {_evidence_cell(rec)!r}, wanted {ev!r}")
    check(f"diagnosis · {rec['id']}", diagnose(rec).startswith(diag),
          f"got {diagnose(rec)!r}, wanted prefix {diag!r}")

check("grounded cell: faithful and cited", _grounded_cell(cases[0][0]) == "Yes")
check("grounded cell: unfaithful", _grounded_cell(cases[3][0]) == "No")
check("grounded cell: faithful but uncited",
      _grounded_cell(cases[8][0]) == "Yes, but uncited")
check("grounded cell: unjudged is not scored as failure",
      _grounded_cell({"recall": 1.0}) == "not judged")
check("grounded cell: correct refusal is grounded", _grounded_cell(cases[5][0]) == "Yes")

sc_run = RunResult("B3", {}, [c[0] for c in cases])
sc = simple_scorecard(sc_run)
check("scorecard has the four instructor columns",
      sc.splitlines()[0].count("|") == 5)
check("scorecard has a row per question", len(sc.splitlines()) == len(cases) + 2)
check("scorecard states 'No evidence exists' for unanswerables",
      "No evidence exists" in sc)
check("long questions truncated",
      "…" in simple_scorecard(RunResult("x", {}, [
          {"id": "Z", "question": "q" * 200, "recall": 1.0}])))

fs = failure_summary(sc_run)
check("failures grouped by cause",
      "RETRIEVAL FAULTS" in fs and "GENERATION FAULTS" in fs)
check("clean hits excluded from failures", "clean hit" not in fs)
check("failure summary counts each bucket", "retrieval ·" in fs)
check("empty failure list is explained",
      "No failures recorded" in failure_summary(RunResult("ok", {}, [cases[0][0]])))


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
