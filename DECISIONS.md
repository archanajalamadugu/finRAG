# Decisions

Why this project is shaped the way it is. Written for anyone reading the repo
who wants the reasoning rather than the code.

---

## The use case

An equity research analyst asks factual and comparative questions about
semiconductor company performance — segment revenue, margins, stated risks,
year-over-year change — and gets a cited answer. Every answer names the
company, the filing date and the 10-K Item it came from.

Eight companies, all semiconductors: **NVIDIA, AMD, Intel, Broadcom, Qualcomm,
Micron, Texas Instruments, Applied Materials.** One sector matters — it means
"which of these spent most on R&D" is a meaningful question rather than an
apples-to-oranges one.

## A rejected alternative, and why

The first candidate use case was **Travel Rescue** — air-travel disruption Q&A
over airline contracts of carriage and U.S. DOT regulations, labelling every
claim as a legal right or a voluntary promise.

Rather than argue about feasibility, the decision rule was written first and
then tested: a probe script fetched all twelve candidate sources and reported
which returned real content.

**Two of twelve were usable.** American Airlines returned HTTP 403, United
timed out, and five of six transportation.gov pages refused the request. Only
Delta's contract of carriage and one Federal Register document came through.

The pre-registered verdict said stop, so it stopped. SEC EDGAR is a documented
government API built to be fetched, which removes the largest execution risk
from a short build. The cleaning and chunking code carried over unchanged.

Corpus feasibility is worth testing before committing to a corpus.

## The corpus is defined by ticker, not CIK

EDGAR identifies companies by **CIK**, a permanent number that survives renames
and re-listings. Nobody knows CIKs by heart, and a wrong one fails in the worst
way — you silently get a different company's filings rather than an error.

So `src/fetch_filings.py` lists plain tickers and resolves them at runtime
against the SEC's published mapping. One extra request, no magic numbers in the
source, and an unknown ticker raises instead of quietly shrinking the corpus.

## Why a custom cleaner, when LangChain has HTML loaders

A 10-K breaks a generic loader twice:

- Filings use `<table>` for real financial data **and** for page layout. Flatten
  both and `Data Center 115,186 47,525` arrives with no indication which figure
  belongs to which year.
- EDGAR puts the `$` in its own cell, so a row arrives as
  `['Data Center', '$', '115,186', '$', '47,525']` against a 4-column header.
  Unfixed, the columns stop lining up entirely.

`src/clean.py` classifies tables, folds currency cells back into their values,
and tags every block with its Item section. That section tag is what lets an
answer cite *"NVIDIA 2026-02-25, Item 7"* rather than an opaque chunk number.

One filing needed more: Intel files a **cross-reference 10-K**, where the Item
numbers appear only in an index pointing at page numbers and the body runs
under plain titles. Item-based detection found one section for Intel and 16–19
for everyone else. The cleaner now falls back to reading bare titles when Item
detection finds nothing, which brought Intel to 13 sections.

That's the only custom code in the pipeline. Everything downstream is
off-the-shelf LangChain.

## Why hybrid retrieval

Dense (vector) search matches meaning and is good at *"why did revenue change
year over year"*. It is weakest exactly where financial questions live: exact
strings.

Observed directly on the first real query. Asked for AMD's **Data Center**
segment revenue, vector search returned three AMD passages with no figures in
them; BM25 returned the segment table. "Data Center" is a proper noun made of
two ordinary words, so meaning-based search drifts toward anything discussing
data centres.

The reverse also happened. Asked about AMD's **supply chain** risks, BM25
matched that phrase where it appears incidentally inside a cybersecurity
passage, missing the passage genuinely about supplier constraints.

Keyword search wins when the query phrase labels the answer and loses when the
phrase appears incidentally elsewhere. Vector search has the opposite profile.
Fusing them does not eliminate the failure — it makes it rarer.

## Why the answer prompt has three exits

ANSWER, CLARIFY, or REFUSE. The third was written first.

The worst output here is not "I don't know" — it is a confident, well-formatted,
plausible number that is not in the filing, because an analyst can act on it.

The **attribution rule** was added after watching it fail. Asked about AMD's
supply chain risks, the system answered using Intel's and NVIDIA's passages
while citing a third AMD passage that said nothing of the kind. The claim was
even true of AMD in the real world; it simply was not supported by the source
it pointed at. Eight competitors describe similar risks in similar language,
so similarity had to be ruled out as evidence explicitly.

## Where the evaluation questions live

In the notebook, section 12 (`TEST_QUESTIONS`). Eight questions: four factual,
one cross-company, one ambiguous, two unanswerable. They are deliberately kept
next to the code that runs them rather than in a separate file that can drift
out of sync with it.
