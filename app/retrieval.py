# retrieval.py
# Sits between the vector store and the LLM.
# Takes the user's question, searches for matching chunks, and glues
# them together into one string that we can drop into the prompt.

from app.vector_store import VectorStore
from app.config import TOP_K
from app import document_registry as registry

def retrieve_context(
    vector_store: VectorStore,
    query: str,
    top_k: int = TOP_K,
    doc_ids: list[str] | None = None,
) -> tuple[str, list[dict]]:
    results = vector_store.search(query, top_k=top_k, doc_ids=doc_ids)
    if not results:
        return "", []

    sources = []
    context_parts = []

    for i, r in enumerate(results):
        doc_info = registry.get_document(r["doc_id"])
        filename = doc_info["original_name"] if doc_info else "Unknown Document"
        source_id = i + 1

        # Add metadata for the UI
        r["filename"] = filename
        r["source_id"] = source_id
        sources.append(r)

        # Format the chunk for the LLM prompt with explicit citation markers
        context_parts.append(f"[Source {source_id}: {filename}]\n{r['text']}")

    context = "\n\n---\n\n".join(context_parts)
    return context, sources
