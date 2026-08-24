# extras — an earlier from-scratch implementation

Before the Week 2 sessions were reviewed, this project implemented chunking,
hybrid retrieval, reranking and a full metric suite by hand. It works and it is
tested (`python3 -m extras.test_pipeline`, 167 tests).

It is **not** what the notebook runs, for two reasons:

1. The sessions taught LangChain + a vector DB, and said repeatedly not to
   build these pieces from scratch. `EnsembleRetriever` does in three lines
   what `retrieve.py` does in three hundred.
2. Evaluation is bonus credit this week. A formal Recall@5 / MRR / nDCG suite
   is Week 4 material, and the sessions explicitly said not to build one now.

Kept because the reasoning inside is still the reasoning behind the pipeline —
why hybrid retrieval, why RRF fuses on rank rather than score, why table
handling has to be held apart from boundary placement.

The live path is `src/` + `notebooks/FinRAG_Colab.ipynb`.
