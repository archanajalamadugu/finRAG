# FinRAG — video demo script

**646 spoken words**, plus roughly 30 seconds of waiting for the app to answer.

Be honest with yourself about the timing: at a normal speaking pace that lands
at **4:50–5:00**, which is the ceiling with no margin. Time one read-through
before you record. If you come in over 4:50, use the cut list near the bottom —
it's there because you will probably need it, not as a formality.

Read it aloud once regardless. Anywhere the words feel unlike you, change them
— you have to sound like you understand it, and you do.

---

## Before you hit record

- [ ] Notebook run top to bottom, finishing at section 13 so the app is live
- [ ] **Delete cells 52 and 53 first** (see "Notebook cleanup" at the bottom) — otherwise Run All ends with an unstyled duplicate of your app
- [ ] Gradio open in a browser tab, dropdown on **All companies**, chat history cleared
- [ ] `finrag_architecture.png` open in Preview, second window
- [ ] Notebook scrolled to section 12 (the scorecard) in a third window
- [ ] Notifications off, Slack and Mail closed

Switch between three windows only: **browser → Preview → VS Code.**

---

## 0:00 – 0:22 · What it is · *68 words*

> *[SCREEN: the Gradio app, nothing typed]*

"This is FinRAG. It answers questions about eight semiconductor companies —
NVIDIA, AMD, Intel, Broadcom, Qualcomm, Micron, Texas Instruments and Applied
Materials — from their latest annual reports filed with the SEC.

A 10-K is the couple-hundred-page report every US public company files each
year. Analysts read them to compare companies, and that reading is the job this
automates. One sector on purpose, so comparison questions mean something."

---

## 0:22 – 0:55 · How it works · *117 words*

> *[SCREEN: the architecture diagram]*

"Two halves.

The green half runs once. Download eight filings, clean the HTML, split the
text into chunks, turn each chunk into a vector — a list of numbers capturing
its meaning. Eight thousand of those go into Pinecone.

The blue half runs on every question. The question becomes a vector using the
same model — it has to be the same one, or the numbers aren't comparable. Two
searches run side by side: vector search for meaning, keyword search for exact
strings like segment names. Results merge, a reranker picks the best five, and
the model answers from those five and nothing else — it isn't remembering these
companies, it's reading five passages I handed it."

---

## 0:55 – 2:10 · The demo · *135 words + 3 pauses*

> *[SCREEN: the app]*

"Three questions, three different behaviours."

**Type:** `What was AMD's Data Center segment revenue?`

> *[wait ~2s]*

"A figure, its unit, and a citation naming the filing and the section — so you
can open the real 10-K and check it. I checked this one by hand: 16,635 million
against 12,579 the year before, exact. Scale matters — a table says 'in
millions' once at the top, and the row is meaningless without it."

**Set the dropdown to `AMD`, then type:** `What supply chain risks are disclosed?`

> *[wait]*

"The dropdown isn't decoration. Picking AMD filters the search to AMD's
passages *before* it runs. All eight companies describe near-identical risks in
near-identical language, so without it the system pulls Intel's paragraph and
presents it as AMD's. It did exactly that — I'll come back to it."

**Type:** `What is NVIDIA's forecast revenue for fiscal 2028?`

> *[wait]*

"It refuses — and names why. Forward-looking guidance isn't in a 10-K. For a
financial tool, a confident wrong number is far worse than no number."

---

## 2:10 – 3:00 · The two comparisons · *115 words*

> *[SCREEN: notebook, section 9]*

"Two comparisons were required.

**Chunking.** Same text split two ways — fixed 800-character chunks, and
semantic chunking, which splits where the meaning shifts. Identical input to
both, so the comparison was fair.

Fixed won two of four, semantic one, one tie. The sophisticated technique lost,
and I can say why: semantic chunks came out three times longer, and a long
chunk covers several topics, so its vector is a blur instead of a sharp signal.
Semantic won the one open-ended question and lost the pointed factual ones —
which is most of what you ask a 10-K.

> *[SCREEN: section 10]*

**Reranking.** A reranker reads the question and each passage together and
rescores them. On-company passages went from four out of five to five out of
five; latency went from 1.15 seconds to 2.74. That's the trade, and I'd take
it."

---

## 3:00 – 4:05 · Evaluation, and the failure worth showing · *135 words*

> *[SCREEN: notebook, section 12 — the scorecard]*

"Eight test questions. Five correct, three failures. What matters is that every
failure is labelled either **retrieval** — wrong passages came back — or
**generation** — right passages came back and the answer was still wrong. Those
need opposite fixes, so guessing wrong makes it worse.

All three of mine are retrieval. Zero generation faults — now.

Asked about AMD's supply chain risks, it retrieved AMD's real risk passage,
then answered from Intel's and NVIDIA's, citing an AMD passage that said
nothing of the kind. The claim was even true of AMD in the real world — it just
wasn't in the source it pointed at. That's the exact failure citations exist to
prevent.

I fixed it with an attribution rule in the prompt: another company's passage
isn't weaker evidence, it's no evidence. And fixing it *revealed* a retrieval
fault underneath, which the borrowed text had been hiding."

---

## 4:05 – 4:35 · Close · *76 words*

> *[SCREEN: back to the app]*

"What I'd fix next: three questions still fail on retrieval — diagnosed, not
solved. And eight test questions isn't enough for a real system.

What I'll take away is that every metric here could be improved by making the
system worse. Table recall goes up if you retrieve only tables. The scorecard
shows zero faults if the questions are easy. The number never meant much on its
own — reading what actually came back did.

Thanks."

---

## If you need to cut to stay under 5:00

Drop in this order — each is self-contained, so removing it leaves no gap:

1. The "Scale matters — a table says 'in millions'…" sentence (–25 words)
2. The reranking paragraph, keeping only "reranking took on-company passages
   from four out of five to five, and latency from 1.15 seconds to 2.74"
   (–45 words)
3. The final sentence of the evaluation section, "And fixing it *revealed* a
   retrieval fault underneath…" (–20 words)
4. In the close, "Table recall goes up if you retrieve only tables. The
   scorecard shows zero faults if the questions are easy." (–20 words)

Don't cut the refusal demo or the misattribution story. Those two are what
distinguish this from a tutorial.

---

## Notebook cleanup before recording

Three fixes in `notebooks/FinRAG.ipynb`:

1. **Delete cell 52** — an older, unstyled copy of the Gradio app sitting
   *after* the styled one. On Run All it replaces your pastel UI with the plain
   version. Click the cell, then the bin icon in its toolbar.
2. **Delete cell 53** — empty.
3. **Cell 11 (markdown)** says "four `.html` files". Change **four** to
   **eight** — left over from when the corpus was smaller.

A stale sentence and a duplicate app cell are exactly what gets noticed if the
grader opens the notebook.

---

## If something breaks on camera

Keep recording and say what happened. An unbothered "that's the share link
expiring, the local one still works" reads as competence; restarting over a
hiccup costs more than the hiccup does.

The likely one: **Gradio share links expire after 72 hours.** If you recorded
earlier and came back, re-run section 13 for a fresh link, or use the local
`127.0.0.1` URL — same app, never expires.
