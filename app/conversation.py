# conversation.py
# Manages persistent conversation history using SQLite.
# Each conversation holds a sequence of user/assistant messages,
# along with metadata like the model used and the context chunks retrieved.
#
# Conversations are auto-titled from the first user question and can be
# listed, loaded, renamed, and deleted. All data stays local.

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from app.config import DATA_DIR
import os

DB_PATH = os.path.join(DATA_DIR, "conversations.db")


def _get_connection() -> sqlite3.Connection:
    """Get a connection to the conversations database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create the conversations and messages tables if they don't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id  TEXT PRIMARY KEY,
            title            TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id       TEXT PRIMARY KEY,
            conversation_id  TEXT NOT NULL,
            role             TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content          TEXT NOT NULL,
            context_chunks   TEXT NOT NULL DEFAULT '[]',
            model_used       TEXT NOT NULL DEFAULT '',
            timestamp        TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_msg_conv
        ON messages(conversation_id, timestamp)
    """)
    conn.commit()
    conn.close()


def _now() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _auto_title(question: str) -> str:
    """Generate a short title from the first question."""
    title = question.strip().replace("\n", " ")
    if len(title) > 60:
        title = title[:57] + "..."
    return title


# -- conversation CRUD ------------------------------------------------------

def create_conversation(title: str = "New Conversation") -> str:
    """Create a new conversation and return its ID."""
    conv_id = uuid.uuid4().hex
    now = _now()
    conn = _get_connection()
    conn.execute(
        "INSERT INTO conversations (conversation_id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return conv_id


def list_conversations() -> list[dict]:
    """Return all conversations, newest first."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str) -> dict | None:
    """Get a single conversation by ID."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def rename_conversation(conversation_id: str, title: str) -> bool:
    """Rename a conversation. Returns True if it existed."""
    conn = _get_connection()
    cursor = conn.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
        (title, _now(), conversation_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its messages. Returns True if it existed."""
    conn = _get_connection()
    cursor = conn.execute(
        "DELETE FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def clear_all_conversations():
    """Delete all conversations and messages."""
    conn = _get_connection()
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()


# -- message CRUD -----------------------------------------------------------

def add_message(
    conversation_id: str,
    role: str,
    content: str,
    context_chunks: list[str] | None = None,
    model_used: str = "",
) -> str:
    """Add a message to a conversation and return its ID."""
    msg_id = uuid.uuid4().hex
    now = _now()
    chunks_json = json.dumps(context_chunks or [])

    conn = _get_connection()
    conn.execute(
        "INSERT INTO messages "
        "(message_id, conversation_id, role, content, context_chunks, model_used, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, chunks_json, model_used, now),
    )
    # update the conversation's updated_at timestamp
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
        (now, conversation_id),
    )
    conn.commit()
    conn.close()
    return msg_id


def get_messages(conversation_id: str) -> list[dict]:
    """Get all messages in a conversation, ordered chronologically."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()

    messages = []
    for r in rows:
        msg = dict(r)
        # parse context_chunks from JSON string back to list
        msg["context_chunks"] = json.loads(msg.get("context_chunks", "[]"))
        messages.append(msg)
    return messages


def get_message_count(conversation_id: str) -> int:
    """Count how many messages are in a conversation."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


# initialise the database on import
init_db()
