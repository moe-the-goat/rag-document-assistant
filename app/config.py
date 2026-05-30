# config.py
# All the settings for the project live here in one place.
# You can override any of them with environment variables if needed.

import os


# -- Ollama connection --
# where Ollama is running and which models we're using
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# -- Chunking --
# 1000 chars per chunk with 200 overlap balances keeping enough context
# per chunk while not overflowing the model's effective context window.
# These values work well with 4096-8192 token context windows.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# -- Paths --
# figure out where the project root is, then put runtime stuff under data/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# -- Retrieval --
# how many chunks to pull from the vector store when answering a question
# 8 gives broad coverage without overwhelming the model's context window
TOP_K = int(os.getenv("TOP_K", "8"))

# -- Security --
# API key authentication (optional): if not set, auth is disabled
# Set this to any secret string to require authentication on all endpoints
API_KEY = os.getenv("API_KEY", "")

# maximum upload file size in megabytes
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# rate limiting: max requests per minute for the /ask endpoint
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

# -- Logging --
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# make sure the data folders exist so we don't have to check later
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
