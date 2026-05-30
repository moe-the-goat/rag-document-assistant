# main.py
# This is the FastAPI app -- the backend that ties everything together.
# It exposes endpoints for uploading documents, asking questions,
# checking status, managing documents, and clearing the store.

import os
import shutil
import uuid
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import UPLOADS_DIR, OLLAMA_BASE_URL
from app.models import (
    QuestionRequest, AnswerResponse, UploadResponse,
    StatusResponse, DocumentInfo,
)
from app.ingestion import ingest_document
from app.vector_store import VectorStore
from app.retrieval import retrieve_context
from app.llm import generate_answer
from app import document_registry as registry


# -- app setup ---------------------------------------------------------------

app = FastAPI(
    title="AI Document Assistant",
    description=(
        "A privacy-first Retrieval-Augmented Generation system that answers "
        "questions grounded in the content of uploaded documents. "
        "All processing happens locally — no data ever leaves the machine."
    ),
    version="2.0.0",
)

# let the Streamlit frontend (or anything else) talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# one shared vector store for all requests
vector_store = VectorStore()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".html"}


# -- routes ------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    # just a quick check to see if the server is alive
    return {"status": "running", "service": "AI Document Assistant (RAG)"}


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    # take a file from the user, parse it, chunk it, embed it, store it
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # save with a random name so we don't get collisions or path tricks
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOADS_DIR, safe_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # -- duplicate detection via SHA-256 hash --------------------------------
    file_hash = registry.compute_file_hash(file_path)
    existing = registry.find_duplicate(file_hash)
    if existing:
        # same content already uploaded, clean up and let the user know
        os.remove(file_path)
        raise HTTPException(
            status_code=409,
            detail=(
                f"This document has already been uploaded as "
                f"'{existing['original_name']}'. Duplicate files are "
                f"skipped to keep the knowledge base clean."
            ),
        )

    # -- ingestion pipeline --------------------------------------------------
    try:
        chunks = ingest_document(file_path)
        file_size = os.path.getsize(file_path)

        # register in the document registry first so we have a doc_id
        doc_id = registry.register_document(
            original_name=file.filename,
            file_hash=file_hash,
            file_type=ext,
            file_size=file_size,
            chunk_count=len(chunks),
        )

        # add chunks to vector store tagged with this doc_id
        vector_store.add_chunks(chunks, doc_id=doc_id)

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        # something went wrong, clean up the file we saved
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        filename=file.filename,
        doc_id=doc_id,
        chunks_added=len(chunks),
        total_chunks=vector_store.total_chunks,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask_question(req: QuestionRequest):
    # this is where the RAG magic happens:
    # 1) find relevant chunks  2) build prompt  3) ask the LLM
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    context, context_chunks = retrieve_context(
        vector_store, question, doc_ids=req.doc_ids,
    )

    answer = generate_answer(context, question, req.model)

    return AnswerResponse(answer=answer, context_chunks=context_chunks)


@app.get("/models", response_model=list[str], tags=["Models"])
def get_available_models():
    # fetch available models dynamically from Ollama API
    # filter out embedding-only models so users only see chat models
    EMBEDDING_PATTERNS = {"embed", "embedding"}
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            chat_models = []
            for m in data.get("models", []):
                name = m["name"].lower()
                # skip models that are clearly embedding-only
                if any(pat in name for pat in EMBEDDING_PATTERNS):
                    continue
                chat_models.append(m["name"])
            if chat_models:
                return chat_models
    except Exception:
        pass
    return ["qwen3:4b"]  # fallback


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
def list_documents():
    # return all registered documents with their metadata
    docs = registry.list_documents()
    return [DocumentInfo(**d) for d in docs]


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    # delete a single document: remove its chunks from the vector store
    # and its entry from the registry
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    removed = vector_store.delete_by_doc_id(doc_id)
    registry.delete_document(doc_id)

    return {
        "message": f"Document '{doc['original_name']}' deleted.",
        "chunks_removed": removed,
        "total_chunks": vector_store.total_chunks,
    }


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def get_status():
    # tell the caller how many chunks we have and if we're ready
    return StatusResponse(
        total_chunks=vector_store.total_chunks,
        status="ready" if vector_store.total_chunks > 0 else "empty",
    )


@app.post("/clear", tags=["Documents"])
def clear_store():
    # nuke everything: the vector index, the registry, and all uploaded files
    vector_store.clear()
    registry.clear_all()
    for fname in os.listdir(UPLOADS_DIR):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return {"message": "All documents, embeddings, and registry cleared."}
