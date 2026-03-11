# vector_store.py
# This wraps the FAISS index so the rest of the app doesn't have to deal
# with raw vectors.  You just call add_chunks() and search() and it handles
# embedding, indexing, and saving to disk behind the scenes.
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

    def __init__(self):
        self.embeddings = get_embedding_model()
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []
        self.dimension: int | None = None
        self._load_if_exists()

    def _index_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "faiss.index")

    def _chunks_path(self) -> str:
        return os.path.join(VECTOR_DB_DIR, "chunks.npy")

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

    def _save(self):
        # write everything to disk so it survives a restart
        # same temp dir workaround here for the FAISS unicode path bug
        if self.index is not None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_index = os.path.join(tmp, "faiss.index")
                faiss.write_index(self.index, tmp_index)
                shutil.copy2(tmp_index, self._index_path())
            np.save(self._chunks_path(), np.array(self.chunks, dtype=object))

    # -- adding data ---------------------------------------------------------

    def add_chunks(self, chunks: list[str]):
        # take a list of text chunks, embed them, and add to the index
        # if the index doesn't exist yet we create it here
        if not chunks:
            return

        vectors = self.embeddings.embed_documents(chunks)
        vectors_np = np.array(vectors, dtype="float32")

        if self.index is None:
            self.dimension = vectors_np.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(vectors_np)
        self.chunks.extend(chunks)
        self._save()

    def clear(self):
        # wipe everything -- the in-memory index and the files on disk
        self.index = None
        self.chunks = []
        self.dimension = None
        for path in [self._index_path(), self._chunks_path()]:
            if os.path.exists(path):
                os.remove(path)

    # -- searching -----------------------------------------------------------

    def search(self, query: str, top_k: int = 4) -> list[str]:
        # embed the query and find the closest chunks by L2 distance
        # returns empty list if nothing's been indexed yet
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.embeddings.embed_query(query)
        query_np = np.array([query_vector], dtype="float32")

        k = min(top_k, self.index.ntotal)
        _, indices = self.index.search(query_np, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
