"""Offline stand-in for the Nebius embedding API, so chunking/retrieval logic
can be tested without network. Same call signature as the real embedder."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class MockEmbedder:
    def __init__(self, dim=64, seed=0):
        self.dim = dim
        self.seed = seed
        self._fitted = False

    def fit(self, corpus):
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   min_df=1, sublinear_tf=True)
        X = self.vec.fit_transform(corpus)
        k = max(2, min(self.dim, X.shape[1] - 1, max(2, X.shape[0] - 1)))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        self.svd.fit(X)
        self._fitted = True
        return self

    def embed_documents(self, texts):
        if not self._fitted:
            self.fit(texts)
        X = self.vec.transform(texts)
        V = self.svd.transform(X).astype(np.float32)
        n = np.linalg.norm(V, axis=1, keepdims=True)
        return V / (n + 1e-9)

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, texts):
        return self.embed_documents(texts)
