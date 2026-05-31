import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Create a temporary directory for tests
TEST_DIR = tempfile.mkdtemp()
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
TEST_UPLOADS_DIR = os.path.join(TEST_DATA_DIR, "uploads")
TEST_VECTOR_DB_DIR = os.path.join(TEST_DATA_DIR, "vector_db")

os.makedirs(TEST_UPLOADS_DIR, exist_ok=True)
os.makedirs(TEST_VECTOR_DB_DIR, exist_ok=True)

# Important: We must patch the config BEFORE importing the app modules that use it
patch("app.config.DATA_DIR", TEST_DATA_DIR).start()
patch("app.config.UPLOADS_DIR", TEST_UPLOADS_DIR).start()
patch("app.config.VECTOR_DB_DIR", TEST_VECTOR_DB_DIR).start()
patch("app.config.API_KEY", "").start()  # Disable auth for most tests

import app.document_registry as registry  # noqa: E402
import app.conversation as conv  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402

# Patch the database paths in the modules
registry.DB_PATH = os.path.join(TEST_DATA_DIR, "registry.db")
conv.DB_PATH = os.path.join(TEST_DATA_DIR, "conversations.db")

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Create test environment and clean up after all tests finish."""
    # Ensure tables are created
    registry.init_db()
    conv.init_db()

    yield

    # Cleanup after all tests
    shutil.rmtree(TEST_DIR, ignore_errors=True)

@pytest.fixture(autouse=True)
def clean_db():
    """Clear databases and vector store before each test."""
    registry.clear_all()
    conv.clear_all_conversations()

    # Clear uploads dir and vector db dir
    for d in [TEST_UPLOADS_DIR, TEST_VECTOR_DB_DIR]:
        for fname in os.listdir(d):
            fpath = os.path.join(d, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

    yield

@pytest.fixture
def test_client():
    """FastAPI test client."""
    return TestClient(fastapi_app)

@pytest.fixture
def mock_vector_store():
    """Mock the vector store to avoid FAISS/BM25 initialization and LLM embedding calls during tests."""
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.total_chunks = 0
        mock_vs.add_chunks.return_value = None
        mock_vs.delete_by_doc_id.return_value = 5  # arbitrary
        yield mock_vs

@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock Ollama embeddings to avoid hitting the API in all tests."""
    with patch("langchain_ollama.OllamaEmbeddings.embed_documents") as mock_embed_docs, \
         patch("langchain_ollama.OllamaEmbeddings.embed_query") as mock_embed_query:
        # Return a fake vector for each input document (e.g., 768 dimensions)
        mock_embed_docs.side_effect = lambda docs: [[0.1] * 768 for _ in docs]
        mock_embed_query.return_value = [0.1] * 768
        yield mock_embed_docs

@pytest.fixture
def mock_llm():
    """Mock the LLM generation and embedding to run instantly without Ollama."""
    with patch("app.main.generate_answer_stream", return_value=["This is a mocked answer from the AI."]) as mock_gen, \
         patch("app.main.retrieve_context", return_value=("Mocked context.", ["Mocked context."])) as mock_ret, \
         patch("app.main.ingest_document", return_value=["Mock chunk"] * 5) as mock_ingest:
        yield {
            "generate": mock_gen,
            "retrieve": mock_ret,
            "ingest": mock_ingest
        }
