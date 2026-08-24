# Setup — running FinRAG locally in VS Code

Everything here happens once. No code is written; you're getting the workshop
ready.

Commands go in **VS Code's built-in terminal**: open the project folder in VS
Code, then **Terminal → New Terminal** from the menu bar. It opens at the
bottom, already pointed at your project folder.

---

## 1 · Check your Python

```bash
python3 --version
```

You want **3.10 or newer**. If it prints something older, install a current
version from [python.org/downloads](https://www.python.org/downloads/).

A note on why we won't use the Python that ships with macOS: Apple installs one
for the operating system's own use, and installing packages into it can break
system tools. That's a large part of why the next step exists.

---

## 2 · Create a virtual environment

A virtual environment — "venv" — is a private box of libraries belonging to
this project alone. Without one, every Python project on your Mac shares a
single pile of packages, and two projects wanting different versions of the
same library break each other. This is the habit that prevents that.

```bash
python3 -m venv .venv
```

That makes a `.venv` folder. Now **activate** it:

```bash
source .venv/bin/activate
```

Your terminal prompt should now start with `(.venv)`. That's how you know
you're inside the box.

> **You re-activate every time you open a new terminal.** If a command fails
> with "module not found", the first thing to check is whether `(.venv)` is
> still on your prompt. It's the single most common stumble, and it isn't a
> sign anything is broken.

---

## 3 · Install the libraries

```bash
pip install -r requirements.txt
```

This reads `requirements.txt` and fetches everything listed. It takes a few
minutes and prints a lot. Warnings in yellow are normal; only red **ERROR**
lines matter.

---

## 4 · Get your two API keys

An API key is a long password proving a request came from you, so usage lands
on your account. You generate these yourself — nobody sends you one.

**Nebius** — supplies both AI models.

1. Go to <https://tokenfactory.nebius.com/project/api-keys>
2. Sign in with the account holding your credits
3. Click **Create API key**
4. Copy it immediately — it is shown **once**

**Pinecone** — the vector database.

1. Go to <https://app.pinecone.io>
2. Sign in, then open **API Keys** in the left sidebar
3. Copy the default key, or create one

> If Nebius says "AI Studio" rather than "Token Factory", you're in the right
> place — they renamed the product.

---

## 5 · Put the keys in `.env`

```bash
cp .env.example .env
```

Open `.env` in VS Code and paste each key in, replacing the placeholder text.
No quotes, no spaces around the `=`:

```
NEBIUS_API_KEY=your_actual_key_here
PINECONE_API_KEY=your_actual_key_here
```

`.env` is listed in `.gitignore`, so Git ignores it completely and it can never
reach your public repo. `.env.example` stays — it has no secrets and it tells
anyone else what keys the project needs.

**Never paste a key into a notebook cell.** Cell output is saved into the
`.ipynb` file, and that file goes to GitHub.

---

## 6 · Point VS Code at your environment

VS Code needs telling to use the Python inside `.venv` rather than your system
one.

1. Open `notebooks/FinRAG.ipynb`
2. Top right of the notebook, click **Select Kernel**
3. Choose **Python Environments…**
4. Pick the one showing `.venv` — usually marked *Recommended*

If nothing appears, install the **Python** and **Jupyter** extensions from the
Extensions panel (the squares icon in the left sidebar), then try again.

---

## 7 · Prove it works

In the notebook, run **cell 1** (the ▶ appears when you hover over a cell).

It should print `found` for both keys. Then run **cell 2** — the model list.
That's the first real proof your Nebius key works.

If cell 1 says MISSING: `.env` isn't in the project's top folder, or a name is
misspelled — they must match exactly, capitals and underscores included.

---

## Where the filings come from

You don't download them. Cell 4 does.

SEC EDGAR is the US government's public archive of company filings, with an
open API built to be fetched. `src/fetch_filings.py` does two lookups: first it resolves each **ticker** to
a **CIK** (a permanent SEC-assigned ID) using the SEC's published mapping file,
then it asks for that company's filing history, picks the newest `10-K`, and
saves it into `data/raw/`.

Eight companies, all semiconductors — one sector, so comparison questions are
meaningful.

Doing it in code rather than by hand is deliberate: it's repeatable, it always
gets the current newest filing, and anyone who clones your repo gets the same
corpus without you emailing them anything.

The SEC asks API users to identify themselves in each request; the `UA` line in
`src/fetch_filings.py` does that. It's a courtesy requirement of their
fair-access policy and it's already set.

`data/` is gitignored, so the filings never bloat your repo.

---

## Everyday commands

```bash
source .venv/bin/activate    # start of every new terminal
git add -A                   # stage your changes
git commit -m "message"      # save them locally
git push                     # send them to GitHub
```
