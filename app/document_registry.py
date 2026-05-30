# document_registry.py
# Manages a lightweight SQLite database that tracks every uploaded document.
# This gives us per-document metadata, duplicate detection via SHA-256 hashing,
# and the ability to list/delete individual documents without wiping everything.
#
# SQLite is perfect here: it's file-based, zero-config, built into Python,
# and keeps all data local -- no external database server needed.

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from app.config import DATA_DIR


DB_PATH = os.path.join(DATA_DIR, "registry.db")


def _get_connection() -> sqlite3.Connection:
    """Get a connection to the registry database, creating it if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # so we get dict-like rows
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create the documents table if it doesn't exist yet."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id         TEXT PRIMARY KEY,
            original_name  TEXT NOT NULL,
            file_hash      TEXT NOT NULL,
            file_type      TEXT NOT NULL,
            file_size      INTEGER NOT NULL,
            file_path      TEXT NOT NULL DEFAULT '',
            chunk_count    INTEGER NOT NULL DEFAULT 0,
            uploaded_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)
    """)
    # migration: add file_path column if upgrading from an older schema
    try:
        conn.execute("ALTER TABLE documents ADD COLUMN file_path TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's content for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def find_duplicate(file_hash: str) -> dict | None:
    """Check if a document with this hash already exists. Returns it or None."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def register_document(
    original_name: str,
    file_hash: str,
    file_type: str,
    file_size: int,
    file_path: str,
    chunk_count: int,
) -> str:
    """Register a new document and return its doc_id."""
    doc_id = uuid.uuid4().hex
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO documents (doc_id, original_name, file_hash, file_type, file_size, file_path, chunk_count, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, original_name, file_hash, file_type, file_size, file_path,
         chunk_count, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return doc_id


def list_documents() -> list[dict]:
    """Return all registered documents, newest first."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    """Get a single document by its ID."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def delete_document(doc_id: str) -> bool:
    """Delete a document from the registry. Returns True if it existed."""
    conn = _get_connection()
    cursor = conn.execute(
        "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def clear_all():
    """Remove all documents from the registry."""
    conn = _get_connection()
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()


# initialise the database on import so the table is always ready
init_db()
