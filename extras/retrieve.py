"""
Retrieval: dense, sparse, and the fusion of the two.

Why hybrid, concretely
----------------------
Dense retrieval is good at "what does Intel say about its foundry business"
and bad at "Data Center". Embeddings compress meaning, and the thing a
financial question most often hinges on is an exact string -- a segment name,
a fiscal-year label, a figure. BM25 is the opposite: it cannot paraphrase,
but it never fumbles a literal. Running both and fusing the ranks gets the
strengths without having to guess in advance which kind of question arrived.

Why Reciprocal Rank Fusion and not score blending
-------------------------------------------------
Cosine similarity and BM25 scores are not on the same scale and their
distributions move with the query. Normalising them against each other means
picking a weight, and any weight we pick is tuned on the eval set we are
about to report -- which quietly invalidates the number. RRF only reads
*positions*, so there is nothing to tune and nothing to leak. It is the
honest default for a project that has to report its own retrieval quality.

Why an in-memory numpy index rather than Chroma
------------------------------------------------
The corpus is four filings -- a few thousand chunks. Exact cosine over a
(n, d) float32 matrix is a single matmul and takes milliseconds at this size,
so an approximate-nearest-neighbour store buys nothing measurable while
adding a moving dependency to a notebook that has to run first time on
someone else's machine. `ChromaDenseIndex` implements the same interface for
anyone who wants the vector DB; `NumpyDenseIndex` is the default because at
this scale the simpler thing is also the faster thing.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------- tokenisation

# Keep decimals, commas inside numbers, and percent signs attached: "47,525"
# and "12.4%" must survive as single tokens or BM25 loses exactly the strings
# it was brought in to catch.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d[\d,\.]*%?")

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "these", "those", "as", "with", "from", "we", "our", "us",
}


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    toks = [t.lower().rstrip(".") for t in _TOKEN_RE.findall(text or "")]
    toks = [t for t in toks if t]
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOP]
    return toks


# ------------------------------------------------------------------- result

@dataclass
class Hit:
    """One retrieved chunk plus how it got here."""
    idx: int                       # position in the indexed chunk list
    score: float
    source: str                    # "dense" | "sparse" | "rrf" | "rerank"
    rank: int = 0
    components: Dict[str, int] = None   # per-retriever rank, for RRF hits

    def __post_init__(self):
        if self.components is None:
            self.components = {}


# -------------------------------------------------------------- dense index

class NumpyDenseIndex:
    """Exact cosine similarity over an in-memory matrix."""

    def __init__(self):
        self.mat: Optional[np.ndarray] = None

    def add(self, vectors: np.ndarray) -> "NumpyDenseIndex":
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {v.shape}")
        # Normalise once at index time so query time is a plain dot product.
        self.mat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return self

    def search(self, qvec: np.ndarray, k: int = 20) -> List[Hit]:
        if self.mat is None or self.mat.shape[0] == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.mat @ q
        k = min(k, sims.shape[0])
        # argpartition for the top-k, then sort just those.
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [Hit(idx=int(i), score=float(sims[i]), source="dense", rank=r + 1)
                for r, i in enumerate(top)]

    def __len__(self) -> int:
        return 0 if self.mat is None else int(self.mat.shape[0])


class ChromaDenseIndex:
    """
    Same interface, backed by Chroma. Provided for parity with the plan;
    not the default. Chroma's own ids are positional strings so results map
    straight back onto the chunk list.
    """

    def __init__(self, collection_name: str = "finrag", client=None):
        import chromadb
        self.client = client or chromadb.EphemeralClient()
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.col = self.client.create_collection(
            collection_name, metadata={"hnsw:space": "cosine"})
        self.n = 0

    def add(self, vectors: np.ndarray, batch: int = 5000) -> "ChromaDenseIndex":
        v = np.asarray(vectors, dtype=np.float32)
        for s in range(0, v.shape[0], batch):
            sl = v[s:s + batch]
            self.col.add(ids=[str(s + i) for i in range(sl.shape[0])],
                         embeddings=sl.tolist())
        self.n = int(v.shape[0])
        return self

    def search(self, qvec: np.ndarray, k: int = 20) -> List[Hit]:
        if self.n == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        r = self.col.query(query_embeddings=[q.tolist()], n_results=min(k, self.n))
        ids, dists = r["ids"][0], r["distances"][0]
        return [Hit(idx=int(i), score=1.0 - float(d), source="dense", rank=rk + 1)
                for rk, (i, d) in enumerate(zip(ids, dists))]

    def __len__(self) -> int:
        return self.n


# ------------------------------------------------------------- sparse index

class BM25Index:
    """
    Okapi BM25, implemented here rather than imported.

    Thirty lines with no dependency, deterministic, and testable offline --
    which matters because BM25 is doing real work in this pipeline and a
    silent tokenisation change in someone else's package would move the
    numbers we are about to report. k1=1.5 / b=0.75 are the standard
    defaults and are deliberately left alone: tuning them on the eval set
    would leak the test set into the retriever.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: List[List[str]] = []
        self.tf: List[Counter] = []
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.doclen: np.ndarray = np.zeros(0)
        self.avgdl: float = 0.0
        self.postings: Dict[str, List[int]] = {}

    def add(self, texts: Sequence[str]) -> "BM25Index":
        self.docs = [tokenize(t) for t in texts]
        self.tf = [Counter(d) for d in self.docs]
        self.doclen = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.doclen.mean()) if len(self.docs) else 0.0
        self.postings = {}
        for i, c in enumerate(self.tf):
            for term in c:
                self.df[term] += 1
                self.postings.setdefault(term, []).append(i)
        n = len(self.docs)
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at zero
        # so a term appearing in almost every document cannot score negative
        # and drag an otherwise good document down.
        self.idf = {t: max(0.0, math.log(1.0 + (n - d + 0.5) / (d + 0.5)))
                    for t, d in self.df.items()}
        return self

    def search(self, query: str, k: int = 20) -> List[Hit]:
        if not self.docs:
            return []
        q = tokenize(query)
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for term in q:
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i in self.postings[term]:
                f = self.tf[i][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doclen[i] / (self.avgdl + 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        nz = int((scores > 0).sum())
        if nz == 0:
            return []
        k = min(k, nz)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(idx=int(i), score=float(scores[i]), source="sparse", rank=r + 1)
                for r, i in enumerate(top)]

    def __len__(self) -> int:
        return len(self.docs)


# ---------------------------------------------------------------------- RRF

def reciprocal_rank_fusion(rankings: Dict[str, List[Hit]],
                           k: int = 60,
                           top_k: int = 20) -> List[Hit]:
    """
    Fuse ranked lists by position only.

    score(d) = sum over retrievers of 1 / (k + rank(d))

    k=60 is the value from the original RRF paper and is left at its default
    for the same reason BM25's k1/b are: any value we chose by looking at our
    own eval set would be a tuned hyperparameter reported as a baseline.
    Larger k flattens the contribution of rank position; at 60 the difference
    between rank 1 and rank 2 is small, which is what we want when neither
    retriever is trusted a priori.
    """
    acc: Dict[int, float] = {}
    comp: Dict[int, Dict[str, int]] = {}
    for name, hits in rankings.items():
        for h in hits:
            acc[h.idx] = acc.get(h.idx, 0.0) + 1.0 / (k + h.rank)
            comp.setdefault(h.idx, {})[name] = h.rank
    order = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Hit(idx=i, score=s, source="rrf", rank=r + 1, components=comp[i])
            for r, (i, s) in enumerate(order[:top_k])]


# --------------------------------------------------------------- retrievers

class HybridRetriever:
    """
    Holds one chunk set and every index built over it.

    `mode` selects the run: "dense", "sparse" or "hybrid". Keeping all three
    behind one object is what makes the experiment matrix cheap -- A1/A2 and
    B1/B2 differ by a string, not by a rebuild.
    """

    def __init__(self,
                 chunks: Sequence[dict],
                 embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
                 dense_index=None,
                 build_sparse: bool = True):
        self.chunks = list(chunks)
        self.texts = [c["text"] for c in self.chunks]
        self.embed_fn = embed_fn
        self.dense = dense_index
        self.sparse = BM25Index().add(self.texts) if build_sparse else None
        self._qcache: Dict[str, np.ndarray] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, chunks: Sequence[dict],
              embed_fn: Callable[[List[str]], np.ndarray],
              dense_backend: str = "numpy",
              vectors: Optional[np.ndarray] = None) -> "HybridRetriever":
        """Embed the chunks (or reuse `vectors`) and build both indexes."""
        texts = [c["text"] for c in chunks]
        vecs = np.asarray(vectors if vectors is not None else embed_fn(texts),
                          dtype=np.float32)
        if vecs.shape[0] != len(texts):
            raise ValueError(
                f"embedder returned {vecs.shape[0]} vectors for {len(texts)} chunks")
        idx = ChromaDenseIndex() if dense_backend == "chroma" else NumpyDenseIndex()
        idx.add(vecs)
        r = cls(chunks, embed_fn=embed_fn, dense_index=idx)
        r.vectors = vecs
        return r

    # -- query -------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = np.asarray(
                self.embed_fn([query]), dtype=np.float32).reshape(-1)
        return self._qcache[query]

    def retrieve(self, query: str, mode: str = "hybrid",
                 top_k: int = 5, per_retriever: int = 20) -> List[Hit]:
        if mode == "dense":
            return self.dense.search(self._embed_query(query), k=top_k)
        if mode == "sparse":
            return self.sparse.search(query, k=top_k)
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}")

        rankings = {
            "dense": self.dense.search(self._embed_query(query), k=per_retriever),
            "sparse": self.sparse.search(query, k=per_retriever),
        }
        return reciprocal_rank_fusion(rankings, top_k=top_k)

    # -- convenience -------------------------------------------------------

    def chunks_for(self, hits: Iterable[Hit]) -> List[dict]:
        return [self.chunks[h.idx] for h in hits]

    def explain(self, query: str, mode: str = "hybrid", top_k: int = 5) -> str:
        """Human-readable top-k, for eyeballing retrieval before trusting it."""
        lines = [f"QUERY  {query}", f"MODE   {mode}", ""]
        for h in self.retrieve(query, mode=mode, top_k=top_k):
            c = self.chunks[h.idx]
            comp = (" via " + ", ".join(f"{k}#{v}" for k, v in sorted(h.components.items()))
                    if h.components else "")
            lines.append(
                f"  {h.rank}. [{h.score:.4f}]{comp}  "
                f"{c.get('company','?')} · {c.get('section','?')} · {c.get('kind','?')}")
            lines.append(f"       {c['text'][:160].replace(chr(10), ' ')}...")
        return "\n".join(lines)


def source_diversity(chunks: Sequence[dict]) -> int:
    """How many distinct companies a result set covers.

    Cross-company questions fail in a specific way: retrieval returns five
    excellent chunks that are all from one filing. Counting distinct sources
    catches that, and no rank metric does.
    """
    return len({c.get("company", "") for c in chunks if c.get("company")})
