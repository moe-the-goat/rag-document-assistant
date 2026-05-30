# vector_store.py
# This wraps the FAISS index so the rest of the app doesn't have to deal
# with raw vectors.  You just call add_chunks() and search() and it handles
# embedding, indexing, and saving to disk behind the scenes.
#
# Each chunk is now tracked with a doc_id so we can delete individual
# documents without wiping the entire index.
#
# One annoying thing: FAISS can't open files if the path has Arabic (or any
# non-ASCII) characters.  Since this project lives in a folder with Arabic
# in the name, we have to copy the index to a temp folder with a clean path
# whenever we read or write it.  Not ideal but it works.

import os
import shutil
import tempfile

import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings

from app.config import EMBEDDING_MODEL, OLLAMA_BASE_URL, VECTOR_DB_DIR


def get_embedding_model() -> OllamaEmbeddings:
    # set up the embedding model that talks to Ollama
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


class VectorStore:
    # manages the FAISS index and keeps the text chunks in sync with it
    # now also tracks which document each chunk belongs to

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []
        self.doc_ids: list[str] = []  # parallel list: doc_id for each chunk
        self.dimension: int | None = None
        self._vectors: np.ndarray | None = None  # keep vectors for rebuild
        self._load_if_exists()

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
        # (the temp dir trick is for the non-ASCII path issue mentioned above)
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
                # old data without doc_ids -- assign "unknown" so nothing breaks
                self.doc_ids = ["unknown"] * len(self.chunks)

            # load raw vectors if they exist (needed for index rebuild on delete)
            vectors_path = self._vectors_path()
            if os.path.exists(vectors_path):
                self._vectors = np.load(vectors_path)
            else:
                # old data -- reconstruct from index
                if self.index is not None and self.index.ntotal > 0:
                    self._vectors = faiss.rev_swig_ptr(
                        self.index.get_xb(), self.index.ntotal * self.index.d
                    ).reshape(self.index.ntotal, self.index.d).copy()
                else:
                    self._vectors = None

    def _save(self):
        # write everything to disk so it survives a restart
        # same temp dir workaround here for the FAISS unicode path bug
        if self.index is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_index = os.path.join(tmp, "faiss.index")
                faiss.write_index(self.index, tmp_index)
                shutil.copy2(tmp_index, self._index_path())
            np.save(self._chunks_path(), np.array(self.chunks, dtype=object))
            np.save(self._doc_ids_path(), np.array(self.doc_ids, dtype=object))
            if self._vectors is not None:
                np.save(self._vectors_path(), self._vectors)

    # -- adding data ---------------------------------------------------------

    def add_chunks(self, chunks: list[str], doc_id: str = "unknown"):
        # take a list of text chunks, embed them, and add to the index
        # if the index doesn't exist yet we create it here
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
        self._save()

    def delete_by_doc_id(self, doc_id: str) -> int:
        # remove all chunks belonging to a specific document
        # since FAISS IndexFlatL2 doesn't support deletion, we rebuild the index
        if self.index is None:
            return 0

        # find which indices to keep
        keep_mask = [did != doc_id for did in self.doc_ids]
        removed_count = keep_mask.count(False)

        if removed_count == 0:
            return 0

        # filter chunks, doc_ids, and vectors
        self.chunks = [c for c, keep in zip(self.chunks, keep_mask) if keep]
        self.doc_ids = [d for d, keep in zip(self.doc_ids, keep_mask) if keep]

        if self._vectors is not None:
            keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
            if keep_indices:
                self._vectors = self._vectors[keep_indices]
            else:
                self._vectors = None

        # rebuild the FAISS index from remaining vectors
        if self.chunks and self._vectors is not None and len(self._vectors) > 0:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self._vectors)
        else:
            self.index = None
            self.dimension = None
            self._vectors = None

        self._save()
        return removed_count

    def clear(self):
        # wipe everything -- the in-memory index and the files on disk
        self.index = None
        self.chunks = []
        self.doc_ids = []
        self.dimension = None
        self._vectors = None
        for path in [self._index_path(), self._chunks_path(),
                     self._doc_ids_path(), self._vectors_path()]:
            if os.path.exists(path):
                os.remove(path)

    # -- searching -----------------------------------------------------------

    def search(self, query: str, top_k: int = 4, doc_ids: list[str] | None = None) -> list[dict]:
        # embed the query and find the closest chunks by L2 distance
        # returns empty list if nothing's been indexed yet
        # optionally filter results to only specific documents
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.embeddings.embed_query(query)
        query_np = np.array([query_vector], dtype="float32")

        # if filtering by doc_ids, we need to search more broadly
        # then filter down to the requested documents
        search_k = min(self.index.ntotal, top_k * 3 if doc_ids else top_k)
        _, indices = self.index.search(query_np, search_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                # if doc_ids filter is set, skip chunks from other documents
                if doc_ids and self.doc_ids[idx] not in doc_ids:
                    continue
                results.append({
                    "text": self.chunks[idx],
                    "doc_id": self.doc_ids[idx],
                })
                if len(results) >= top_k:
                    break
        return results

    def get_chunks_by_doc_id(self, doc_id: str) -> int:
        """Count how many chunks belong to a specific document."""
        return sum(1 for did in self.doc_ids if did == doc_id)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
