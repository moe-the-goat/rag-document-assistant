# models.py
# Pydantic models for the API request and response bodies.
# Keeping these separate from main.py so things stay organized
# and we can import them from other places without pulling in FastAPI.

from typing import Any
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    # what the user sends when they ask a question
    question: str
    model: str = "qwen2.5:latest"
    doc_ids: list[str] | None = None  # optional: filter to specific documents
    conversation_id: str | None = None  # optional: save to a conversation


class AnswerResponse(BaseModel):
    # what we send back: the answer + the chunks we used to make it
    answer: str
    context_chunks: list[str]
    conversation_id: str | None = None  # the conversation this was saved to


class UploadResponse(BaseModel):
    # confirmation after a document is uploaded and processed
    filename: str
    doc_id: str
    chunks_added: int
    total_chunks: int


class StatusResponse(BaseModel):
    # quick overview of how much data is in the store
    total_chunks: int
    status: str


class DocumentInfo(BaseModel):
    # metadata for a single document in the registry
    doc_id: str
    original_name: str
    file_type: str
    file_size: int
    chunk_count: int
    uploaded_at: str


class ConversationInfo(BaseModel):
    # summary of a conversation for the list view
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageInfo(BaseModel):
    # a single message in a conversation
    message_id: str
    conversation_id: str
    role: str
    content: str
    context_chunks: list[Any] = []
    model_used: str = ""
    timestamp: str


class ConversationDetail(BaseModel):
    # full conversation with all its messages
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageInfo]


class RenameRequest(BaseModel):
    # request to rename a conversation
    title: str
