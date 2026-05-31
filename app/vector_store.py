# vector_store.py
# This wraps the FAISS index and BM25 keyword index so the rest of the app
# doesn't have to deal with raw vectors or tokenization.
#
# HYBRID SEARCH: We combine two search methods for better results:
#   - FAISS (semantic): understands meaning ("revenue" matches "annual income")
#   - BM25 (keyword): catches exact matches (dates, names, serial numbers)
# Results are fused using Reciprocal Rank Fusion (RRF), a proven technique
# that merges ranked lists from different retrieval methods.
#
# Each chunk is tracked with a doc_id so we can delete individual
# documents without wiping the entire index.
#
# FAISS unicode path workaround: FAISS can't handle non-ASCII paths,
# so we copy index files to a temp dir for read/write operations.

import os
import re
import shutil
import tempfile

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

from app.config import EMBEDDING_MODEL, OLLAMA_BASE_URL, VECTOR_DB_DIR


def get_embedding_model() -> OllamaEmbeddings:
    # set up the embedding model that talks to Ollama
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25: lowercase, split on word boundaries.
    Works for both English and Arabic text."""
    return re.findall(r"\w+", text.lower())


class VectorStore:
    """Manages FAISS (semantic) + BM25 (keyword) hybrid search with
    per-document chunk tracking and Reciprocal Rank Fusion."""

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []
        self.doc_ids: list[str] = []  # parallel list: doc_id for each chunk
        self.dimension: int | None = None
        self._vectors: np.ndarray | None = None  # keep vectors for rebuild
        self._bm25: BM25Okapi | None = None  # keyword search index
        self._load_if_exists()

    @property
    def re_ranker(self) -> CrossEncoder:
        if not hasattr(self, "_re_ranker"):
            # Load a lightweight, highly accurate cross-encoder model locally
            self._re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self._re_ranker

    # -- file paths ----------------------------------------------------------

    def _index_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "faiss.index")

    def _chunks_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "chunks.npy")

    def _doc_ids_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "doc_ids.npy")

    def _vectors_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "vectors.npy")

    # -- saving and loading --------------------------------------------------

    def _load_if_exists(self):
        # if we already have saved data from a previous run, load it back
        index_path = self._index_path()
        chunks_path = self._chunks_path()
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_index = os.path.join(tmp, "faiss.index")
                shutil.copy2(index_path, tmp_index)
                self.index = faiss.read_index(tmp_index)
            self.dimension = self.index.d
            self.chunks = np.load(chunks_path, allow_pickle=True).tolist()

            # load doc_ids if they exist (backward compatibility with old data)
            doc_ids_path = self._doc_ids_path()
            if os.path.exists(doc_ids_path):
                self.doc_ids = np.load(doc_ids_path, allow_pickle=True).tolist()
            else:
                self.doc_ids = ["unknown"] * len(self.chunks)

            # load raw vectors if they exist (needed for index rebuild on delete)
            vectors_path = self._vectors_path()
            if os.path.exists(vectors_path):
                self._vectors = np.load(vectors_path)
            else:
                if self.index is not None and self.index.ntotal > 0:
                    self._vectors = faiss.rev_swig_ptr(
                        self.index.get_xb(), self.index.ntotal * self.index.d
                    ).reshape(self.index.ntotal, self.index.d).copy()
                else:
                    self._vectors = None

            # rebuild BM25 index from loaded chunks
            self._rebuild_bm25()

    def _save(self):
        # write everything to disk so it survives a restart
        if self.index is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_index = os.path.join(tmp, "faiss.index")
                faiss.write_index(self.index, tmp_index)
                shutil.copy2(tmp_index, self._index_path())
            np.save(self._chunks_path(), np.array(self.chunks, dtype=object))
            np.save(self._doc_ids_path(), np.array(self.doc_ids, dtype=object))
            if self._vectors is not None:
                np.save(self._vectors_path(), self._vectors)

    def _rebuild_bm25(self):
        """Rebuild the BM25 keyword index from current chunks.
        BM25 doesn't need disk persistence — it's fast to rebuild from text."""
        if self.chunks:
            tokenized = [_tokenize(chunk) for chunk in self.chunks]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    # -- adding data ---------------------------------------------------------

    def add_chunks(self, chunks: list[str], doc_id: str = "unknown"):
        # take a list of text chunks, embed them, and add to both indexes
        if not chunks:
            return

        vectors = self.embeddings.embed_documents(chunks)
        vectors_np = np.array(vectors, dtype="float32")

        if self.index is None:
            self.dimension = vectors_np.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self._vectors = vectors_np
        else:
            self._vectors = np.vstack([self._vectors, vectors_np])

        self.index.add(vectors_np)
        self.chunks.extend(chunks)
        self.doc_ids.extend([doc_id] * len(chunks))
        self._rebuild_bm25()
        self._save()

    def delete_by_doc_id(self, doc_id: str) -> int:
        # remove all chunks belonging to a specific document
        if self.index is None:
            return 0

        keep_mask = [did != doc_id for did in self.doc_ids]
        removed_count = keep_mask.count(False)

        if removed_count == 0:
            return 0

        self.chunks = [c for c, keep in zip(self.chunks, keep_mask) if keep]
        self.doc_ids = [d for d, keep in zip(self.doc_ids, keep_mask) if keep]

        if self._vectors is not None:
            keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
            if keep_indices:
                self._vectors = self._vectors[keep_indices]
            else:
                self._vectors = None

        # rebuild both indexes from remaining data
        if self.chunks and self._vectors is not None and len(self._vectors) > 0:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self._vectors)
        else:
            self.index = None
            self.dimension = None
            self._vectors = None

        self._rebuild_bm25()
        self._save()
        return removed_count

    def clear(self):
        # wipe everything
        self.index = None
        self.chunks = []
        self.doc_ids = []
        self.dimension = None
        self._vectors = None
        self._bm25 = None
        for path in [self._index_path(), self._chunks_path(),
                     self._doc_ids_path(), self._vectors_path()]:
            if os.path.exists(path):
                os.remove(path)

    # -- searching -----------------------------------------------------------

    def _search_faiss(self, query: str, fetch_k: int, doc_ids: list[str] | None = None) -> list[tuple[int, float]]:
        """Semantic search via FAISS. Returns list of (chunk_index, distance)."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.embeddings.embed_query(query)
        query_np = np.array([query_vector], dtype="float32")

        k = min(fetch_k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.chunks):
                if doc_ids and self.doc_ids[idx] not in doc_ids:
                    continue
                results.append((int(idx), float(dist)))
        return results

    def _search_bm25(self, query: str, fetch_k: int, doc_ids: list[str] | None = None) -> list[tuple[int, float]]:
        """Keyword search via BM25. Returns list of (chunk_index, score)."""
        if self._bm25 is None or not self.chunks:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # pair each chunk index with its BM25 score, sort by score descending
        scored = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        # apply doc_ids filter
        if doc_ids:
            scored = [(i, s) for i, s in scored if self.doc_ids[i] in doc_ids]

        return scored[:fetch_k]

    def search(self, query: str, top_k: int = 4, doc_ids: list[str] | None = None) -> list[dict]:
        """
        Hybrid search combining FAISS (semantic) and BM25 (keyword) results
        using Reciprocal Rank Fusion (RRF).

        RRF score for each chunk = sum of 1/(k + rank) across both methods.
        This naturally balances results from both search strategies without
        needing to normalize scores across different scales.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # fetch more candidates from each method for better fusion
        fetch_k = min(top_k * 3, self.index.ntotal)

        faiss_results = self._search_faiss(query, fetch_k, doc_ids)
        bm25_results = self._search_bm25(query, fetch_k, doc_ids)

        # Reciprocal Rank Fusion (k=60 is the standard constant from the paper)
        RRF_K = 60
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # sort by combined RRF score (highest first)
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # Stage 2: Re-ranking with Cross-Encoder
        top_candidates = sorted_indices[:fetch_k]
        if not top_candidates:
            return []

        # Prepare pairs of (query, chunk_text) for the cross-encoder
        cross_inp = [[query, self.chunks[idx]] for idx in top_candidates]
        cross_scores = self.re_ranker.predict(cross_inp)

        # Pair indices with their cross-encoder scores
        reranked = [(idx, score) for idx, score in zip(top_candidates, cross_scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in reranked[:top_k]:
            results.append({
                "text": self.chunks[idx],
                "doc_id": self.doc_ids[idx],
                "chunk_idx": idx,
                "score": float(score)
            })
        return results

    def get_chunks_by_doc_id(self, doc_id: str) -> int:
        """Count how many chunks belong to a specific document."""
        return sum(1 for did in self.doc_ids if did == doc_id)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
