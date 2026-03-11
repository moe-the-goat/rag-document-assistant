# config.py
# All the settings for the project live here in one place.
# You can override any of them with environment variables if needed.

import os


# -- Ollama connection --
# where Ollama is running and which models we're using
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# -- Chunking --
# 800 chars per chunk with 150 overlap seemed like a good balance
# between keeping chunks focused and not losing context at the edges
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# -- Paths --
# figure out where the project root is, then put runtime stuff under data/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# -- Retrieval --
# how many chunks to pull from the vector store when answering a question
TOP_K = int(os.getenv("TOP_K", "6"))

# make sure the data folders exist so we don't have to check later
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
