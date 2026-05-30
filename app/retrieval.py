# retrieval.py
# Sits between the vector store and the LLM.
# Takes the user's question, searches for matching chunks, and glues
# them together into one string that we can drop into the prompt.

from app.vector_store import VectorStore
from app.config import TOP_K


def retrieve_context(
    vector_store: VectorStore,
    query: str,
    top_k: int = TOP_K,
    doc_ids: list[str] | None = None,
) -> tuple[str, list[str]]:
    # grab the most relevant chunks and join them with a separator
    # returns (context_string, list_of_chunk_texts) 
    # optionally filter to specific documents via doc_ids
    results = vector_store.search(query, top_k=top_k, doc_ids=doc_ids)
    if not results:
        return "", []

    chunk_texts = [r["text"] for r in results]
    context = "\n\n---\n\n".join(chunk_texts)
    return context, chunk_texts
