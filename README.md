# AI Document Assistant -- Retrieval-Augmented Generation (RAG)

A question-answering system that lets users upload documents and ask
natural-language questions about their content.  Instead of relying
solely on the language model's training data, the system retrieves
relevant passages from the uploaded documents and uses them as context
when generating answers.  This approach -- known as Retrieval-Augmented
Generation -- produces responses that are grounded in the actual
document text.

Everything runs **locally** using [Ollama](https://ollama.com).
No external API keys or paid services are required.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Running the Application](#running-the-application)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Documentation](#documentation)

---

## System Architecture

```
+---------------------------------------------------------------+
|                       User Interface                          |
|                (Streamlit Chat  /  REST API)                  |
+---------------+-----------------------+-----------------------+
                |                       |
          Upload Documents         Ask Questions
                |                       |
                v                       v
+----------------------------+  +-------------------------------+
|    Ingestion Pipeline      |  |     Retrieval Pipeline        |
|                            |  |                               |
|  1. Parse document         |  |  5. Embed the question        |
|     (PDF / TXT)            |  |  6. Search FAISS index        |
|  2. Split into chunks      |  |  7. Fetch top-K chunks        |
|  3. Generate embeddings    |  |  8. Build prompt with context |
|  4. Store in FAISS         |  |  9. Send to LLM               |
+------------+---------------+  +---------------+---------------+
             |                                  |
             v                                  v
+----------------------------+  +-------------------------------+
|   FAISS Vector Database    |<-|   Ollama LLM  (qwen3:4b)     |
|   (persisted to disk)      |  |   + nomic-embed-text          |
+----------------------------+  +-------------------------------+
```

### Pipeline Summary

1. The user uploads a PDF or plain-text file.
2. The file is parsed and split into overlapping text chunks (800 characters
   with 150-character overlap by default).
3. Each chunk is converted into a high-dimensional vector using the
   `nomic-embed-text` embedding model served by Ollama.
4. The vectors are stored in a FAISS flat-L2 index that is persisted to disk.
5. When the user asks a question, the question text is embedded with the same
   model.
6. FAISS performs a nearest-neighbour search and returns the top-K most
   similar chunks.
7. Those chunks are injected into a prompt template along with the original
   question.
8. The prompt is sent to the `qwen3:4b` language model running on Ollama.
9. The model generates an answer that is strictly grounded in the retrieved
   context.

---

## Tech Stack

| Component         | Technology              | Role                                       |
|-------------------|-------------------------|---------------------------------------------|
| Language          | Python 3.11+            | All backend and frontend code               |
| Orchestration     | LangChain               | Text splitting, embedding and LLM wrappers  |
| LLM               | Ollama  (qwen3:4b)      | Answer generation                           |
| Embeddings        | Ollama  (nomic-embed-text)| Vectorising text chunks and queries        |
| Vector Database   | FAISS  (faiss-cpu)       | Storing and searching embeddings            |
| API Framework     | FastAPI + Uvicorn        | REST backend                                |
| Frontend          | Streamlit                | Browser-based chat interface                |

---

## Project Structure

```
.
|-- app/
|   |-- __init__.py          Package metadata
|   |-- config.py            Centralised settings (env-var overridable)
|   |-- ingestion.py         Document parsing and text chunking
|   |-- vector_store.py      FAISS index wrapper with persistence
|   |-- retrieval.py         Semantic retrieval logic
|   |-- llm.py               Ollama LLM client and prompt template
|   |-- models.py            Pydantic request / response schemas
|   +-- main.py              FastAPI application and route handlers
|
|-- ui/
|   +-- app.py               Streamlit chat frontend
|
|-- data/
|   |-- uploads/             Uploaded files (runtime, git-ignored)
|   +-- vector_db/           FAISS index files  (runtime, git-ignored)
|
|-- docs/
|   +-- DOCUMENTATION.md     Detailed technical documentation
|
|-- requirements.txt         Pinned Python dependencies
|-- .gitignore
+-- README.md                This file
```

---

## Prerequisites

- **Python 3.11** or newer
- **Ollama** installed and running (https://ollama.com)

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd "AI Document Assistant using Retrieval-Augmented"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the required Ollama models

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

---

## Running the Application

You need two terminal windows: one for the backend, one for the frontend.

### Terminal 1 -- FastAPI backend

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**.
Interactive Swagger documentation is at **http://localhost:8000/docs**.

### Terminal 2 -- Streamlit frontend

```bash
python -m streamlit run ui/app.py
```

The UI will open in your browser at **http://localhost:8501**.

### Using the application

1. Open http://localhost:8501 in your browser.
2. Use the sidebar on the left to upload a PDF or TXT file and click
   **Ingest Document**.
3. Once the document is processed, type a question in the chat box at the
   bottom.
4. The system will retrieve relevant passages and generate an answer.

---

## API Reference

| Method | Endpoint  | Description                                    |
|--------|-----------|------------------------------------------------|
| GET    | /         | Health check                                   |
| POST   | /upload   | Upload and ingest a document (PDF or TXT)      |
| POST   | /ask      | Ask a question about the ingested documents    |
| GET    | /status   | Return the number of chunks in the store       |
| POST   | /clear    | Delete all documents and embeddings            |

### Upload a document

```bash
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"What is the main topic of the document?\"}"
```

---

## Configuration

All settings are defined in `app/config.py` and can be overridden with
environment variables:

| Variable          | Default                    | Description                          |
|-------------------|----------------------------|--------------------------------------|
| OLLAMA_BASE_URL   | http://localhost:11434      | Ollama server address                |
| LLM_MODEL         | qwen3:4b                   | Model used for answer generation     |
| EMBEDDING_MODEL   | nomic-embed-text           | Model used for text embeddings       |
| CHUNK_SIZE        | 800                        | Max characters per text chunk        |
| CHUNK_OVERLAP     | 150                        | Overlap between consecutive chunks   |
| TOP_K             | 6                          | Number of chunks retrieved per query |

---

## Documentation

For a detailed explanation of every module, design decision, and the full
RAG pipeline, see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).
