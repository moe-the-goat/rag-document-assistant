# Technical Documentation

## AI Document Assistant -- Retrieval-Augmented Generation

This document provides an in-depth explanation of every component in the
project, the rationale behind key design decisions, and how all the pieces
fit together into a working RAG pipeline.

---

## Table of Contents

1. [What is Retrieval-Augmented Generation?](#1-what-is-retrieval-augmented-generation)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Module-by-Module Walkthrough](#3-module-by-module-walkthrough)
   - 3.1 [config.py -- Centralised Configuration](#31-configpy----centralised-configuration)
   - 3.2 [ingestion.py -- Document Parsing and Chunking](#32-ingestionpy----document-parsing-and-chunking)
   - 3.3 [vector_store.py -- FAISS Index Wrapper](#33-vector_storepy----faiss-index-wrapper)
   - 3.4 [retrieval.py -- Semantic Retrieval](#34-retrievalpy----semantic-retrieval)
   - 3.5 [llm.py -- Language Model Integration](#35-llmpy----language-model-integration)
   - 3.6 [models.py -- Data Schemas](#36-modelspy----data-schemas)
   - 3.7 [main.py -- FastAPI Application](#37-mainpy----fastapi-application)
   - 3.8 [ui/app.py -- Streamlit Frontend](#38-uiapppy----streamlit-frontend)
4. [Data Flow: End-to-End Example](#4-data-flow-end-to-end-example)
5. [Design Decisions and Trade-offs](#5-design-decisions-and-trade-offs)
6. [Technology Choices](#6-technology-choices)
7. [Known Limitations](#7-known-limitations)
8. [Possible Future Improvements](#8-possible-future-improvements)

---

## 1. What is Retrieval-Augmented Generation?

Large Language Models (LLMs) are trained on vast amounts of text, but they
have two significant weaknesses when used for question-answering in a
specific domain:

- **Knowledge cut-off:** They do not know about documents that were not in
  their training data.
- **Hallucination:** When uncertain, they may generate plausible-sounding
  but incorrect answers.

Retrieval-Augmented Generation (RAG) addresses both problems by combining
an information retrieval step with the language model.  Before the model
generates an answer, the system searches a knowledge base (in our case, the
uploaded documents) for passages that are relevant to the user's question.
Those passages are injected into the prompt as "context."  The model is then
instructed to answer strictly from that context.

This means:

- The model gets access to information it was never trained on.
- Answers are grounded in actual document text, which drastically reduces
  hallucination.

---

## 2. High-Level Architecture

```
User  --->  Streamlit UI / REST API
                  |
        +---------+---------+
        |                   |
   Upload file        Ask a question
        |                   |
        v                   v
 [ingestion.py]      [retrieval.py]
   parse, chunk        embed query
        |               search FAISS
        v                   |
 [vector_store.py]          |
   embed chunks,            v
   store in FAISS     [llm.py]
                       build prompt
                       call Ollama
                            |
                            v
                    Answer (+ source chunks)
```

There are two distinct paths through the system:

1. **Ingestion path** (left side): user uploads a document, the system
   parses it, splits the text, generates embeddings, and stores them.
2. **Query path** (right side): user asks a question, the system embeds
   the question, searches for similar chunks, builds a prompt, and calls
   the LLM.

---

## 3. Module-by-Module Walkthrough

### 3.1 config.py -- Centralised Configuration

**Purpose:** Hold every configurable value in one place.

All settings are read from environment variables with sensible defaults.
This means someone deploying the system on a different machine can change
behaviour (e.g. swap in a larger LLM model) without editing source code.

Key settings:

| Setting         | Default             | Why this value?                            |
|-----------------|---------------------|--------------------------------------------|
| CHUNK_SIZE      | 800 characters      | Small enough for focused retrieval, large enough to retain paragraph-level meaning. |
| CHUNK_OVERLAP   | 150 characters      | Prevents information at chunk boundaries from being lost. |
| TOP_K           | 6                   | Retrieving 6 chunks gives the LLM good coverage without flooding the context window. |
| LLM_MODEL       | qwen3:4b            | A capable 4-billion-parameter model that runs well on consumer hardware. |
| EMBEDDING_MODEL | nomic-embed-text    | High-quality open-source embedding model, optimised for retrieval tasks. |

File paths for runtime data (`data/uploads/`, `data/vector_db/`) are
derived from the project root automatically, and the directories are
created on import if they do not exist.

---

### 3.2 ingestion.py -- Document Parsing and Chunking

**Purpose:** Turn a raw file into a list of text chunks ready for
embedding.

**Step 1 -- Parsing.**
The module supports several formats:

- **Extraction:** `PyMuPDF` reads the uploaded PDF and extracts raw text. Plain text and markdown files are read directly. `python-docx` extracts paragraphs from Word documents, and `BeautifulSoup4` strips tags from HTML files to extract visible text.

The `load_document()` function looks at the file extension to choose the
right parser.

**Step 2 -- Chunking.**
Raw document text can be thousands of characters long.  Embedding models
work best on shorter, focused passages, so we split the text using
LangChain's `RecursiveCharacterTextSplitter`.

How the splitter works:

1. It tries to split on double newlines (paragraph breaks) first.
2. If a chunk is still too large, it falls back to single newlines, then
   sentence-ending periods, then spaces.
3. This recursive strategy means chunks tend to land on natural boundaries
   rather than cutting a sentence in half.

Overlap between chunks (150 characters by default) ensures that
information sitting right at a split boundary is not lost -- it appears in
both the chunk before and the chunk after the boundary.

---

### 3.3 vector_store.py -- FAISS Index Wrapper

**Purpose:** Manage all interactions with FAISS and the parallel chunk
list.

FAISS (Facebook AI Similarity Search) is a library for efficient
nearest-neighbour search in high-dimensional vector spaces.  We use
`IndexFlatL2`, which computes exact L2 (Euclidean) distances between
vectors.  This is the simplest index type; it performs a brute-force scan
over all stored vectors.  For the scale of a single-user document assistant
(hundreds to low thousands of chunks), brute-force is fast enough and
avoids the complexity of approximate indices.

**Embedding generation.**
The `OllamaEmbeddings` class from `langchain-ollama` converts text into
768-dimensional vectors using the `nomic-embed-text` model.  The same model
must be used for both chunk embeddings and query embeddings so that they
share the same vector space.

**Persistence.**
After every write (`add_chunks`), the class saves two files:

- `faiss.index` -- the serialised FAISS index.
- `chunks.npy` -- a NumPy array of the raw text chunks, stored in the same
  order as the corresponding vectors.

On startup, if these files exist, they are loaded back into memory so that
previously uploaded documents remain searchable across server restarts.

**Unicode path workaround.**
FAISS's C++ file I/O layer uses `fopen()` internally, which cannot handle
non-ASCII characters in file paths on Windows.  Since this project is
developed on a system with Arabic characters in the path, we work around
the issue by copying the index file to a temporary directory with a safe
ASCII path before calling `faiss.read_index()` or `faiss.write_index()`.

---

### 3.4 retrieval.py -- Semantic Retrieval

**Purpose:** Bridge between the vector store and the LLM.

This module is intentionally simple: it calls `vector_store.search()` with
the user's query and joins the returned chunks into a single string
separated by `---` dividers.  The result is what gets injected into the
prompt template.

Keeping retrieval in its own module (rather than inlining it in the API
handler) makes it easy to test or replace the retrieval strategy later --
for example, by adding a re-ranking step or filtering by document metadata.

---

### 3.5 llm.py -- Language Model Integration

**Purpose:** Send a prompt to Ollama and return the model's answer.

**Prompt design.**
The prompt template includes three sections:

1. A system instruction telling the model to answer strictly from the
   provided context.
2. The context block (chunks retrieved from the vector store).
3. The user's question.

The instruction explicitly tells the model **not** to make up information.
If the context does not contain the answer, the model should say so.

**qwen3 thinking tags.**
The qwen3 model family has an internal chain-of-thought mode.  When active,
the model wraps its reasoning process in `<think>...</think>` XML tags
before writing the actual answer.  Those tags are not useful for the end
user, so `_strip_thinking_tags()` removes them with a regex before
returning the response.

**Temperature.**
Set to 0.1 (very low) to minimise creative drift.  For a document QA
system, we want deterministic, factual answers rather than varied or
imaginative text.

---

### 3.6 models.py -- Data Schemas

**Purpose:** Define the shape of request and response bodies.

Using Pydantic models gives us:

- Automatic validation of incoming JSON (FastAPI raises 422 if the body
  does not match).
- Auto-generated OpenAPI documentation (visible at `/docs`).
- A single source of truth for the API contract.

The schemas are kept in their own file rather than in `main.py` to avoid
that file growing too large and to allow other parts of the codebase
(tests, CLI tools) to import them without pulling in the FastAPI app.

---

### 3.7 main.py -- FastAPI Application

**Purpose:** Define the HTTP API that exposes the RAG pipeline.

**Endpoints:**

| Endpoint   | What it does                                                |
|------------|-------------------------------------------------------------|
| GET /      | Returns a JSON object confirming the server is alive.       |
| POST /upload | Accepts a file upload, runs the full ingestion pipeline, and returns how many chunks were created. |
| POST /ask  | Accepts a JSON body with a `question` field, runs retrieval + LLM generation, and returns the answer along with the context chunks used. |
| GET /status | Reports how many chunks are in the store and whether it is empty or ready. |
| POST /clear | Deletes all vectors from the FAISS index, removes the persisted files, and deletes uploaded documents from disk. |

**File upload safety.**
Uploaded files are saved with a random UUID filename, not the original
name.  This prevents path-traversal attacks and filename collisions.

**Error handling.**
If ingestion fails (e.g. a corrupt PDF), the endpoint catches the
exception, deletes the file that was just saved, and returns a 500 with a
descriptive message.

**CORS.**
Cross-Origin Resource Sharing is enabled for all origins so that the
Streamlit frontend (which may run on a different port) can talk to the API
without browser-level blocks.

---

### 3.8 ui/app.py -- Streamlit Frontend

**Purpose:** Provide a browser-based interface for non-technical users.

The frontend is split into two areas:

- **Sidebar:** File upload, vector store status, and a button to clear
  everything.
- **Main area:** A chat-style interface where the user types questions
  and sees answers.

All communication with the backend happens through standard HTTP requests
to `http://localhost:8000`.  The UI never touches FAISS or Ollama directly.

The conversation history is stored in `st.session_state` so that it
persists across Streamlit re-runs within the same browser session.  Each
assistant message also stores the retrieved context chunks, which the user
can expand to see exactly which parts of the document were used to generate
the answer.

---

## 4. Data Flow: End-to-End Example

Suppose a user uploads a 10-page PDF about climate change and then asks:
"What are the main causes of global warming?"

### Ingestion phase

1. The PDF is uploaded through the Streamlit sidebar.
2. Streamlit sends a POST request to `/upload` with the file.
3. FastAPI saves the file to `data/uploads/` with a UUID name.
4. `ingestion.py` opens the PDF with PyMuPDF and extracts all text.
5. The text is split into chunks of roughly 800 characters each, with
   150 characters of overlap between consecutive chunks.
6. `vector_store.py` sends the chunks to Ollama's `nomic-embed-text`
   model, which returns a 768-dimensional vector for each chunk.
7. The vectors are added to the FAISS index, and the raw chunk strings
   are appended to the parallel list.
8. Both are saved to disk under `data/vector_db/`.
9. The API returns the number of chunks created.

### Query phase

1. The user types the question in the chat box.
2. Streamlit sends a POST request to `/ask` with the question.
3. `retrieval.py` embeds the question using the same `nomic-embed-text`
   model.
4. FAISS compares the question vector against all stored vectors and
   returns the indices of the 6 closest matches.
5. The corresponding text chunks are fetched from the chunk list and
   concatenated into a "context" string.
6. `llm.py` builds a prompt:
   - System instruction: "Answer only from the context."
   - Context: the 6 retrieved chunks.
   - Question: the user's original question.
7. The prompt is sent to `qwen3:4b` through Ollama.
8. The model generates an answer grounded in those chunks.
9. Any `<think>` tags are stripped from the output.
10. The answer and the context chunks are returned to the frontend.
11. Streamlit displays the answer in the chat and stores the context
    behind an expandable section.

---

## 5. Design Decisions and Trade-offs

### Why Ollama instead of the OpenAI API?

- Zero cost.  Everything runs on the local machine.
- No internet dependency.  The system works offline.
- Data privacy.  Documents never leave the user's computer.
- The trade-off is that local models are smaller and sometimes less
  capable than GPT-4, but for extractive QA over uploaded documents,
  a 4B-parameter model like qwen3 performs well enough.

### Why FAISS instead of a full vector database (Weaviate, Pinecone)?

- FAISS is a single `pip install` with no external services to run.
- For a single-user system with a few thousand chunks, flat-L2 search
  is fast (sub-millisecond) and gives exact results.
- No network round-trips to an external database.
- The trade-off is that FAISS has no built-in metadata filtering or
  multi-tenant support, but we do not need those here.

### Why LangChain?

- LangChain provides well-tested wrappers for text splitting and
  embedding that would take non-trivial effort to write from scratch.
- It abstracts the Ollama HTTP API behind a clean Python interface.
- We use only a small part of LangChain (text splitter + Ollama
  integration), so the dependency is lightweight in practice.

### Why separate the Streamlit UI from the FastAPI backend?

- The UI and the backend have different responsibilities and different
  runtime models (Streamlit re-runs the script on every interaction;
  FastAPI is a long-running ASGI server).
- Keeping them separate means the API can be used without any UI at all
  (e.g. from curl, Postman, or another application).
- It also makes it possible to deploy them on different machines if
  needed.

---

## 6. Technology Choices

| Library / Tool     | Version   | What we use it for                       |
|--------------------|-----------|------------------------------------------|
| Python             | 3.11+     | Primary language for the entire project  |
| FastAPI            | 0.115.0   | REST API framework                       |
| Uvicorn            | 0.30.6    | ASGI server to run FastAPI               |
| LangChain          | 0.3.25    | Text splitting utilities                 |
| langchain-ollama   | 0.3.3     | Embedding and chat model wrappers        |
| FAISS (faiss-cpu)  | 1.9.0     | Vector similarity search                 |
| PyMuPDF (fitz)     | 1.26.7    | PDF text extraction                      |
| python-docx        | 1.1.2     | Word document text extraction            |
| BeautifulSoup4     | 4.12.3    | HTML text extraction                     |
| Streamlit          | 1.45.0    | Frontend chat interface                  |
| Ollama             | (local)   | Serves LLM and embedding models locally  |

---

## 7. Known Limitations

- **PDF layout sensitivity.**  PyMuPDF extracts text in reading order, but
  complex layouts (multi-column, tables, figures) may produce garbled text.
- **No OCR.**  Scanned PDFs (images of text) will produce empty extractions
  because there is no OCR step.
- **Single-user.**  The FAISS index and chunk list are loaded into memory
  once.  Concurrent uploads from multiple users could cause race conditions.
- **No document-level metadata.**  Once chunks are in the store, there is
  no way to filter by document or delete a single document without clearing
  everything.
- **Model quality.**  A 4B-parameter model may struggle with highly
  technical or nuanced questions compared to larger models.

---

## 8. Possible Future Improvements

- Implement per-document metadata so users can query specific files.
- Add a re-ranking step (e.g. a cross-encoder) after FAISS retrieval to
  improve relevance.
- Switch to an approximate nearest-neighbour index (e.g. IVF) for better
  performance at larger scales.
- Add authentication to the API.
- Add automated tests for each module.
- Add a dropdown to the Streamlit UI to dynamically switch between downloaded Ollama models via the backend.
