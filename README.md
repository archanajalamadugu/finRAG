# FinRAG — asking questions of SEC filings

Week 2 project · The Gen Academy, Mastering Agentic AI Bootcamp.

A RAG app that answers analyst questions about eight semiconductor companies
from their latest 10-K filings, cites every claim, and refuses when the filings
don't contain the answer.

NVIDIA · AMD · Intel · Broadcom · Qualcomm · Micron · Texas Instruments ·
Applied Materials. One sector, so comparison questions are meaningful.

## What it does

```
SEC EDGAR  →  clean the HTML (keep tables whole, tag Item sections)
           →  chunk it TWO ways: fixed-size vs semantic
           →  embed → Pinecone
           →  retrieve: vector search + BM25, merged
           →  rerank with a cross-encoder → best 5
           →  answer with citations, or decline
```

Everything after the cleaning step is stock LangChain, used the way Session 1
demoed it. Runs locally in VS Code — see **[SETUP.md](SETUP.md)**, then open
`notebooks/FinRAG.ipynb`.

## The two things Project 2 asks for

**1 · Compare two chunking strategies.** Fixed-size windows
(`RecursiveCharacterTextSplitter`, 800/100) against semantic boundaries
(`SemanticChunker` at the 90th percentile), on the same questions through the
same retriever.

Tables are kept whole by *both* strategies. That matters: if only one protected
tables, a difference in results couldn't be attributed to boundary placement
rather than table handling. Holding it constant is what makes the comparison
mean anything.

**2 · Measure the reranking step.** Same index, same retrieval, cross-encoder
on and off — reported with timings, because a reranker that buys better
passages for two extra seconds is a trade-off, not a free win.

## Why hybrid retrieval, specifically here

Dense (vector) search matches on meaning and is excellent at *"why did revenue
change year over year"*. It is weakest exactly where financial questions live:
exact strings. Ask for **Data Center** and dense search drifts to any passage
about data centres; BM25 goes straight to the literal term.

Segment names, fiscal-year labels and figures are the substance of a 10-K
question, so both retrievers earn their place. `EnsembleRetriever` merges them
with reciprocal rank fusion — whatever ranks high in *both* lists wins.

## Why keep a custom cleaner

A 10-K isn't an ordinary web page, and a generic HTML loader breaks on it twice:

- Filings use `<table>` for real financial data **and** for page layout. A
  generic loader flattens both, so `Data Center 115,186 47,525` arrives with no
  indication which figure belongs to which year.
- EDGAR puts the `$` in its own cell, so a row comes through as
  `['Data Center', '$', '115,186', '$', '47,525']` against a 4-column header.
  Unfixed, the columns stop lining up entirely.

`src/clean.py` classifies tables, folds currency cells back into their values,
and tags every block with its Item section — which is what lets an answer cite
*"NVIDIA 2025-02-26, Item 7"* instead of an opaque chunk number.

That's the only custom code in the pipeline. Everything downstream is
off-the-shelf.

## The corpus is defined by ticker, not by CIK

EDGAR identifies companies by **CIK** — a permanent SEC-assigned number that
survives renames and re-listings. But nobody knows CIKs by heart, and a wrong
one fails in the worst possible way: you silently get a different company's
filings rather than an error.

So `src/fetch_filings.py` lists plain tickers and resolves them at runtime
against [the SEC's own published mapping](https://www.sec.gov/files/company_tickers.json).
One extra request, no magic numbers in the source, and an unknown ticker raises
instead of quietly shrinking the corpus.

It is also precisely the lookup an "add any company" feature needs, so that
stretch goal is already half-built — `fetch_corpus()` takes a `companies`
override for exactly this.

## Evaluation

Bonus credit this week — Week 4 covers it properly, and the sessions were
explicit that a metrics framework isn't wanted now.

So the deliverable is a four-column table: question · did the right evidence
come back · is the answer grounded and cited · what happened. Plus at least one
failure diagnosed as **retrieval** ("the wrong passages came back") or
**generation** ("it had good passages and wrote a bad answer"). Those have
completely different fixes, and telling them apart is the point.

`src/scorecard.py` produces both. It's ~80 lines, deliberately.

## Layout

```
src/fetch_filings.py   ticker → CIK → the latest 10-K for each company
src/clean.py           HTML → clean, section-tagged, table-aware blocks
src/to_documents.py    those blocks → LangChain Documents (the only bridge)
src/prompts.py         the answer prompt, and the refusal path
src/scorecard.py       the evaluation table + retrieval-vs-generation diagnosis
WRITEUP.md             the full project write-up — comparisons, evaluation, iterations
finrag_architecture.png the pipeline diagram WRITEUP.md refers to
SETUP.md               one-time local setup: venv, keys, VS Code kernel
DECISIONS.md           why the project is shaped this way
ENGINEERING_LOG.md     incidents recorded while they were still fresh
notebooks/FinRAG.ipynb the notebook you actually run, cell by cell
tests/                 54 offline tests: no network, no API key
extras/                an earlier from-scratch build, kept for reference
```

`extras/` holds a hand-written implementation of chunking, hybrid retrieval,
reranking and a full metric suite (167 further tests). It works, but it isn't
what runs — `EnsembleRetriever` does in three lines what `extras/retrieve.py`
does in three hundred, and the sessions said plainly not to build these from
scratch. Kept because the reasoning inside is still the reasoning behind the
pipeline.

## Running it

Follow [SETUP.md](SETUP.md) once — virtual environment, `pip install`, two API
keys in a `.env` file, and pointing VS Code at the environment. Then open
`notebooks/FinRAG.ipynb` and run the cells top to bottom.

Offline tests, no key and no network:

```bash
python3 -m tests.test_offline      # 54 — fetching, cleaning, tables, sections
python3 -m extras.test_pipeline    # 167 — the reference implementation
```
