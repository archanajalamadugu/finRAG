# Engineering log

Problems hit while building FinRAG, how each was diagnosed, and what changed.
Written as they happened rather than reconstructed afterwards, because the
diagnosis is the interesting part and it's the first thing you forget.

---

## 1 · Imports broke because a library shipped a major version

**Symptom.** None yet — caught before running anything.

**How it surfaced.** Reading the output of `pip install -r requirements.txt`.
Two package names in the list didn't match expectations:

- `langchain-1.3.16` — the requirements said `langchain>=0.3.0`, so 0.3.x was
  expected. Version **1.x** meant a major release had landed, and by convention
  a major version bump is how maintainers signal deliberate breaking changes.
- `langchain-classic-1.0.8` — never requested, so something pulled it in as a
  dependency. The name is the giveaway: "classic" and "legacy" are what
  libraries call the package where old furniture gets stored after a
  reorganisation.

Major version plus a package literally named "classic" implies things moved.
The pipeline imports `EnsembleRetriever`, `ContextualCompressionRetriever` and
`CrossEncoderReranker` — exactly the sort of long-standing pieces that get
relocated.

**Verification.** Checked LangChain's own reference documentation rather than
trusting the inference: all three now live under `langchain_classic`. Then
confirmed empirically with a script importing all 16 things the notebook
needs, because docs lag and what matters is what's actually installed.

**Fix.** Try the new path, fall back to the old one:

```python
try:
    from langchain_classic.retrievers import EnsembleRetriever   # LangChain 1.x
except ImportError:
    from langchain.retrievers import EnsembleRetriever           # LangChain 0.3
```

Then pinned `requirements.txt` to `langchain>=1.0.0` with `langchain-classic`
listed explicitly, so anyone cloning the repo gets a set of versions that work
together.

**Root cause.** `>=0.3.0` was too loose. An open-ended version range is a bet
that no library will ever break anything, and libraries break things.

**Lesson.** Read install output rather than scrolling past it. An unexpected
package name is a real signal. And verify a hunch against a primary source
before acting on it.

---

## 2 · A side effect lost during a rewrite

**Symptom.** Would have been `ModuleNotFoundError: No module named 'src'` at
the cleaning cell.

**How it surfaced.** Not from output — from tracing through the code before
running it. VS Code starts a notebook with its working directory set to the
notebook's *own* folder, so Python would be standing in `notebooks/` while
`src` sits one level up in the project root.

**Root cause.** The original Colab version had a cell that unzipped the project
and then called `os.chdir(ROOT)`. Rewriting for local VS Code, that cell was
deleted — there was no zip to unpack any more — and the `chdir` went with it.
A cell doing two jobs, removed while thinking about only one of them.

**Fix.** Walk *up* from wherever the notebook starts until the folder
containing `src` is found, then work from there:

```python
ROOT = pathlib.Path.cwd()
while not (ROOT / 'src').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
```

Solved in the code rather than in an editor setting, so it behaves the same
for anyone who opens the project in any editor.

**A correction worth recording.** The original diagnosis also predicted the
`.env` file wouldn't be found. That part was wrong — `load_dotenv()` with no
arguments searches *upward* through parent directories, so the keys loaded
correctly all along. Python's import system does no such search, which is why
the `src` half of the prediction held and the `.env` half didn't.

**Lesson.** Deleting something that was quietly doing two jobs is a common way
to break things. And a prediction being half right is worth noticing — the
half that was wrong showed a real gap in understanding of how `dotenv` works.

---

## 3 · The planned models didn't exist on this account

**Symptom.** None — caught by listing available models before using any.

**How it surfaced.** The notebook asks the provider what the account can
actually use, rather than hard-coding names, and prints the list. Neither
planned default was present:

| Planned | Available |
|---|---|
| `BAAI/bge-en-icl` (embedding) | `Qwen/Qwen3-Embedding-8B` — the only one |
| `Qwen/Qwen3-235B-A22B` (chat) | `Qwen/Qwen3-235B-A22B-Instruct-2507` and 28 others |

**Fix.** Use what exists. The embedding choice makes itself — there is only
one. For chat, `Qwen/Qwen3-235B-A22B-Instruct-2507` was chosen over the
alternatives because:

- it is an **Instruct** variant, not a *Thinking* one. Reasoning models emit a
  long internal scratchpad before answering, which adds latency to a live demo
  and produces output that needs stripping before it can be parsed.
- strong instruction-following matters more than usual here, because the answer
  prompt requires the model to pick one of three exits (answer / clarify /
  refuse) and refusing correctly is a graded behaviour.
- it is a mixture-of-experts model — 235B total but only ~22B active per token
  — so it is considerably faster than its size suggests.

**Design note that paid off.** The dimension of an embedding model (how many
numbers per vector) has to match the Pinecone index exactly, and a mismatch
fails on every insert. Because the notebook *measures* the dimension from the
model rather than hard-coding it, swapping the embedding model needed no other
change. Hard-coding `1536` would have broken silently here.

**Lesson.** Ask the system what it supports instead of assuming. A list printed
before anything expensive runs costs one API call and removes a whole class of
"why doesn't this work" later.
