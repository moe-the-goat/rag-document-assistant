# models.py
# Pydantic models for the API request and response bodies.
# Keeping these separate from main.py so things stay organized
# and we can import them from other places without pulling in FastAPI.

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    # what the user sends when they ask a question
    question: str


class AnswerResponse(BaseModel):
    # what we send back: the answer + the chunks we used to make it
    answer: str
    context_chunks: list[str]


class UploadResponse(BaseModel):
    # confirmation after a document is uploaded and processed
    filename: str
    chunks_added: int
    total_chunks: int


class StatusResponse(BaseModel):
    # quick overview of how much data is in the store
    total_chunks: int
    status: str
