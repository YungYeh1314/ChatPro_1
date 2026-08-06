"""SQLite 数据层：文档、切片、对话、消息。

初次接触数据库的话，只需要关注这 4 张表：
  documents     原始文档（爬/下载来的每篇内容）
  chunks        文档被切成的块（RAG 检索的最小单位）
  conversations 一次对话
  messages      对话中的每一条消息
"""

import sqlite3
from datetime import datetime

from . import config


def get_conn() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'unknown',
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL REFERENCES documents(id),
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---- documents / chunks ----

def doc_exists(title: str, source: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE title = ? AND source = ?",
            (title, source),
        ).fetchone()
    return row is not None


def insert_document(title: str, source: str, content: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (title, source, content, created_at) VALUES (?, ?, ?, ?)",
            (title, source, content, now()),
        )
        return cur.lastrowid


def insert_chunk(doc_id: int, chunk_index: int, content: str) -> int:
    # token_count 用字符数/2 粗略估算（中文约 2 字符 = 1 token）
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, content, token_count) VALUES (?, ?, ?, ?)",
            (doc_id, chunk_index, content, max(1, len(content) // 2)),
        )
        return cur.lastrowid


# ---- conversations / messages ----

def create_conversation() -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO conversations (created_at) VALUES (?)", (now(),))
        return cur.lastrowid


def add_message(conversation_id: int, role: str, content: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now()),
        )
        return cur.lastrowid


def get_messages(conversation_id: int, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    return list(reversed([dict(r) for r in rows]))


def stats() -> dict:
    with get_conn() as conn:
        docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        msgs = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
    return {"documents": docs, "chunks": chunks, "messages": msgs}
