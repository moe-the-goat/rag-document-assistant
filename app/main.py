# main.py
# This is the FastAPI app -- the backend that ties everything together.
# It exposes endpoints for uploading documents, asking questions,
# checking status, and clearing the store.

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import UPLOADS_DIR
from app.models import QuestionRequest, AnswerResponse, UploadResponse, StatusResponse
from app.ingestion import ingest_document
from app.vector_store import VectorStore
from app.retrieval import retrieve_context
from app.llm import generate_answer


# -- app setup ---------------------------------------------------------------

app = FastAPI(
    title="AI Document Assistant",
    description=(
        "A Retrieval-Augmented Generation system that answers questions "
        "grounded in the content of uploaded documents."
    ),
    version="1.0.0",
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

    try:
        chunks = ingest_document(file_path)
        vector_store.add_chunks(chunks)
    except Exception as e:
        # something went wrong, clean up the file we saved
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        filename=file.filename,
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

    context = retrieve_context(vector_store, question)
    context_chunks = context.split("\n\n---\n\n") if context else []

    answer = generate_answer(context, question)

    return AnswerResponse(answer=answer, context_chunks=context_chunks)


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def get_status():
    # tell the caller how many chunks we have and if we're ready
    return StatusResponse(
        total_chunks=vector_store.total_chunks,
        status="ready" if vector_store.total_chunks > 0 else "empty",
    )


@app.post("/clear", tags=["Documents"])
def clear_store():
    # nuke everything: the vector index and all uploaded files
    vector_store.clear()
    for fname in os.listdir(UPLOADS_DIR):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return {"message": "Vector store and uploads cleared."}
