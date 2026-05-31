# main.py
# This is the FastAPI app — the backend that ties everything together.
# It exposes endpoints for uploading documents, asking questions,
# managing conversations, checking status, and clearing the store.
#
# Security features:
#   - Optional API key authentication (set API_KEY env var to enable)
#   - Rate limiting on /ask endpoint (prevents abuse)
#   - File validation (size limits, magic byte checks, filename sanitization)
#   - Structured logging with timestamps for audit trails

import logging
import os
import shutil
import uuid

import requests as http_requests
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import UPLOADS_DIR, OLLAMA_BASE_URL, RATE_LIMIT, LOG_LEVEL
from app.models import (
    QuestionRequest, UploadResponse,
    StatusResponse, DocumentInfo, ConversationInfo,
    ConversationDetail, MessageInfo, RenameRequest,
)
from app.ingestion import ingest_document
from app.vector_store import VectorStore
from app.retrieval import retrieve_context
from app.llm import generate_answer_stream
from app import document_registry as registry
from app import conversation as conv
from app.auth import require_auth
from app.file_validator import validate_file_size, validate_file_type, sanitize_filename


# -- logging setup -----------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag-api")


# -- rate limiter setup ------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# -- app setup ---------------------------------------------------------------

app = FastAPI(
    title="AI Document Assistant",
    description=(
        "A privacy-first Retrieval-Augmented Generation system that answers "
        "questions grounded in the content of uploaded documents. "
        "All processing happens locally — no data ever leaves the machine."
    ),
    version="3.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# one shared vector store for all requests
vector_store = VectorStore()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".html"}


# -- routes: health (no auth required) --------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "service": "AI Document Assistant (RAG)"}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def get_status():
    return StatusResponse(
        total_chunks=vector_store.total_chunks,
        status="ready" if vector_store.total_chunks > 0 else "empty",
    )


# -- routes: documents (auth required) --------------------------------------

@app.post("/upload", response_model=UploadResponse, tags=["Documents"],
          dependencies=[Depends(require_auth)])
async def upload_document(file: UploadFile = File(...)):
    original_name = sanitize_filename(file.filename or "unnamed.pdf")
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # save to disk with a UUID name to prevent collisions
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOADS_DIR, safe_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # -- security validation -------------------------------------------------
    try:
        validate_file_size(file_path)
        validate_file_type(file_path, ext)
    except ValueError as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    # -- duplicate detection -------------------------------------------------
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

    # -- ingestion pipeline --------------------------------------------------
    try:
        chunks = ingest_document(file_path)
        file_size = os.path.getsize(file_path)

        doc_id = registry.register_document(
            original_name=original_name,
            file_hash=file_hash,
            file_type=ext,
            file_size=file_size,
            file_path=file_path,
            chunk_count=len(chunks),
        )

        vector_store.add_chunks(chunks, doc_id=doc_id)
        logger.info(
            f"Document ingested: '{original_name}' → {len(chunks)} chunks "
            f"(doc_id={doc_id})"
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Ingestion failed for '{original_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        filename=original_name,
        doc_id=doc_id,
        chunks_added=len(chunks),
        total_chunks=vector_store.total_chunks,
    )


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"],
         dependencies=[Depends(require_auth)])
def list_documents():
    docs = registry.list_documents()
    return [DocumentInfo(**d) for d in docs]


@app.delete("/documents/{doc_id}", tags=["Documents"],
            dependencies=[Depends(require_auth)])
def delete_document(doc_id: str):
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    removed = vector_store.delete_by_doc_id(doc_id)

    if doc.get("file_path") and os.path.exists(doc["file_path"]):
        os.remove(doc["file_path"])

    registry.delete_document(doc_id)
    logger.info(
        f"Document deleted: '{doc['original_name']}' "
        f"({removed} chunks removed)"
    )

    return {
        "message": f"Document '{doc['original_name']}' deleted.",
        "chunks_removed": removed,
        "total_chunks": vector_store.total_chunks,
    }


# -- routes: Q&A (auth + rate limit) ----------------------------------------

@app.post("/ask", tags=["Q&A"],
          dependencies=[Depends(require_auth)])
@limiter.limit(RATE_LIMIT)
def ask_question(request: Request, req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # conversation management
    conversation_id = req.conversation_id
    if conversation_id:
        if not conv.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation_id = conv.create_conversation(
            title=conv._auto_title(question),
        )

    # save user message
    conv.add_message(
        conversation_id=conversation_id,
        role="user",
        content=question,
        model_used=req.model,
    )

    # RAG pipeline
    context, context_chunks = retrieve_context(
        vector_store, question, doc_ids=req.doc_ids,
    )

    def event_stream():
        # First yield the conversation_id and context_chunks so the UI can update immediately
        init_data = {
            "conversation_id": conversation_id,
            "context_chunks": context_chunks,
        }
        yield f"data: {json.dumps(init_data)}\n\n"

        full_answer = ""
        for chunk in generate_answer_stream(context, question, req.model):
            full_answer += chunk
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

        # After the stream finishes, save the assistant message
        conv.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer,
            context_chunks=context_chunks,
            model_used=req.model,
        )

        # Signal completion
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# -- routes: conversations (auth required) -----------------------------------

@app.get("/conversations", response_model=list[ConversationInfo],
         tags=["Conversations"], dependencies=[Depends(require_auth)])
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


@app.post("/conversations", tags=["Conversations"],
          dependencies=[Depends(require_auth)])
def create_conversation(title: str = "New Conversation"):
    conv_id = conv.create_conversation(title=title)
    return {"conversation_id": conv_id, "title": title}


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail,
         tags=["Conversations"], dependencies=[Depends(require_auth)])
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


@app.patch("/conversations/{conversation_id}", tags=["Conversations"],
           dependencies=[Depends(require_auth)])
def rename_conversation(conversation_id: str, req: RenameRequest):
    if not conv.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    conv.rename_conversation(conversation_id, req.title)
    return {"message": "Conversation renamed.", "title": req.title}


@app.delete("/conversations/{conversation_id}", tags=["Conversations"],
            dependencies=[Depends(require_auth)])
def delete_conversation(conversation_id: str):
    if not conv.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    logger.info(f"Conversation deleted: {conversation_id}")
    return {"message": "Conversation deleted."}


@app.delete("/conversations", tags=["Conversations"],
            dependencies=[Depends(require_auth)])
def clear_all_conversations():
    conv.clear_all_conversations()
    logger.info("All conversations deleted.")
    return {"message": "All conversations deleted."}


# -- routes: models ----------------------------------------------------------

@app.get("/models", response_model=list[str], tags=["Models"])
def get_available_models():
    EMBEDDING_PATTERNS = {"embed", "embedding"}
    try:
        resp = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
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
    return ["qwen2.5:latest"]


# -- routes: maintenance -----------------------------------------------------

@app.post("/clear", tags=["Maintenance"],
          dependencies=[Depends(require_auth)])
def clear_store():
    vector_store.clear()
    registry.clear_all()
    conv.clear_all_conversations()
    for fname in os.listdir(UPLOADS_DIR):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    logger.info("All data cleared (documents, conversations, embeddings)")
    return {"message": "All documents, embeddings, conversations, and registry cleared."}
