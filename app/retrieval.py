# retrieval.py
# Sits between the vector store and the LLM.
# Takes the user's question, searches for matching chunks, and glues
# them together into one string that we can drop into the prompt.

from app.vector_store import VectorStore
from app.config import TOP_K


def retrieve_context(vector_store: VectorStore, query: str, top_k: int = TOP_K) -> str:
    # grab the most relevant chunks and join them with a separator
    # returns empty string if the store has nothing (no docs uploaded yet)
    chunks = vector_store.search(query, top_k=top_k)
    if not chunks:
        return ""
    return "\n\n---\n\n".join(chunks)
