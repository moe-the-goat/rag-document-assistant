# main.py
# This is the FastAPI app -- the backend that ties everything together.
# It exposes endpoints for uploading documents, asking questions,
# managing conversations, checking status, and clearing the store.

import os
import shutil
import uuid
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import UPLOADS_DIR, OLLAMA_BASE_URL
from app.models import (
    QuestionRequest, AnswerResponse, UploadResponse,
    StatusResponse, DocumentInfo, ConversationInfo,
    ConversationDetail, MessageInfo, RenameRequest,
)
from app.ingestion import ingest_document
from app.vector_store import VectorStore
from app.retrieval import retrieve_context
from app.llm import generate_answer
from app import document_registry as registry
from app import conversation as conv


# -- app setup ---------------------------------------------------------------

app = FastAPI(
    title="AI Document Assistant",
    description=(
        "A privacy-first Retrieval-Augmented Generation system that answers "
        "questions grounded in the content of uploaded documents. "
        "All processing happens locally — no data ever leaves the machine."
    ),
    version="3.0.0",
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


# -- routes: health ----------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "service": "AI Document Assistant (RAG)"}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def get_status():
    return StatusResponse(
        total_chunks=vector_store.total_chunks,
        status="ready" if vector_store.total_chunks > 0 else "empty",
    )


# -- routes: documents -------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOADS_DIR, safe_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # duplicate detection
    file_hash = registry.compute_file_hash(file_path)
    existing = registry.find_duplicate(file_hash)
    if existing:
        os.remove(file_path)
        raise HTTPException(
            status_code=409,
            detail=(
                f"This document has already been uploaded as "
                f"'{existing['original_name']}'. Duplicate files are "
                f"skipped to keep the knowledge base clean."
            ),
        )

    # ingestion pipeline
    try:
        chunks = ingest_document(file_path)
        file_size = os.path.getsize(file_path)

        doc_id = registry.register_document(
            original_name=file.filename,
            file_hash=file_hash,
            file_type=ext,
            file_size=file_size,
            file_path=file_path,
            chunk_count=len(chunks),
        )

        vector_store.add_chunks(chunks, doc_id=doc_id)

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        filename=file.filename,
        doc_id=doc_id,
        chunks_added=len(chunks),
        total_chunks=vector_store.total_chunks,
    )


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
def list_documents():
    docs = registry.list_documents()
    return [DocumentInfo(**d) for d in docs]


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    removed = vector_store.delete_by_doc_id(doc_id)

    if doc.get("file_path") and os.path.exists(doc["file_path"]):
        os.remove(doc["file_path"])

    registry.delete_document(doc_id)

    return {
        "message": f"Document '{doc['original_name']}' deleted.",
        "chunks_removed": removed,
        "total_chunks": vector_store.total_chunks,
    }


# -- routes: Q&A -------------------------------------------------------------

@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # if a conversation_id is provided, use it; otherwise create a new one
    conversation_id = req.conversation_id
    if conversation_id:
        # verify it exists
        if not conv.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        # auto-create a conversation titled after the first question
        conversation_id = conv.create_conversation(
            title=conv._auto_title(question),
        )

    # save the user message
    conv.add_message(
        conversation_id=conversation_id,
        role="user",
        content=question,
        model_used=req.model,
    )

    # RAG pipeline: retrieve → generate
    context, context_chunks = retrieve_context(
        vector_store, question, doc_ids=req.doc_ids,
    )
    answer = generate_answer(context, question, req.model)

    # save the assistant message
    conv.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        context_chunks=context_chunks,
        model_used=req.model,
    )

    return AnswerResponse(
        answer=answer,
        context_chunks=context_chunks,
        conversation_id=conversation_id,
    )


# -- routes: conversations ---------------------------------------------------

@app.get("/conversations", response_model=list[ConversationInfo], tags=["Conversations"])
def list_conversations():
    conversations = conv.list_conversations()
    result = []
    for c in conversations:
        result.append(ConversationInfo(
            conversation_id=c["conversation_id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
            message_count=conv.get_message_count(c["conversation_id"]),
        ))
    return result


@app.post("/conversations", tags=["Conversations"])
def create_conversation(title: str = "New Conversation"):
    conv_id = conv.create_conversation(title=title)
    return {"conversation_id": conv_id, "title": title}


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail, tags=["Conversations"])
def get_conversation(conversation_id: str):
    c = conv.get_conversation(conversation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = conv.get_messages(conversation_id)
    return ConversationDetail(
        conversation_id=c["conversation_id"],
        title=c["title"],
        created_at=c["created_at"],
        updated_at=c["updated_at"],
        messages=[MessageInfo(**m) for m in messages],
    )


@app.patch("/conversations/{conversation_id}", tags=["Conversations"])
def rename_conversation(conversation_id: str, req: RenameRequest):
    if not conv.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    conv.rename_conversation(conversation_id, req.title)
    return {"message": "Conversation renamed.", "title": req.title}


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
def delete_conversation(conversation_id: str):
    if not conv.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message": "Conversation deleted."}


# -- routes: models -----------------------------------------------------------

@app.get("/models", response_model=list[str], tags=["Models"])
def get_available_models():
    EMBEDDING_PATTERNS = {"embed", "embedding"}
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            chat_models = []
            for m in data.get("models", []):
                name = m["name"].lower()
                if any(pat in name for pat in EMBEDDING_PATTERNS):
                    continue
                chat_models.append(m["name"])
            if chat_models:
                return chat_models
    except Exception:
        pass
    return ["qwen3:4b"]


# -- routes: maintenance ------------------------------------------------------

@app.post("/clear", tags=["Maintenance"])
def clear_store():
    vector_store.clear()
    registry.clear_all()
    conv.clear_all_conversations()
    for fname in os.listdir(UPLOADS_DIR):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return {"message": "All documents, embeddings, conversations, and registry cleared."}
