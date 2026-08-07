"""SQLite 数据层：文档、切片、对话、消息。

SQLite 是 Python 自带的轻量级关系型数据库，数据保存在一个 .db 文件里。
本文件封装了所有读写 SQLite 的操作，其他模块（ingest、rag、chat_api）
只调用这里的函数，不需要直接写 SQL。

初次接触数据库的话，只需要关注这 4 张表：
  documents     原始文档（下载/导入来的每篇内容）
  chunks        文档被切成的块（RAG 检索的最小单位）
  conversations 一次对话
  messages      对话中的每一条消息
"""

import sqlite3
from datetime import datetime
from . import config


def get_conn() -> sqlite3.Connection:
    """创建并返回一个数据库连接。

    - 确保 data/ 目录存在（不存在就创建；exist_ok=True 表示已存在也不报错）。
    - sqlite3.connect：rag.db 不存在会自动创建空文件，已存在则直接打开。
    - row_factory = sqlite3.Row：让查询结果既支持下标也支持按列名访问，
      这样 row["title"] 比 row[1] 好读得多。
    - PRAGMA foreign_keys = ON：开启外键约束，保证 chunks.doc_id、
      messages.conversation_id 这些引用是有效的。SQLite 默认不检查外键，
      所以每次连接都要手动打开。
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """初始化数据库：创建 4 张表（如果还不存在）。

    CREATE TABLE IF NOT EXISTS 是幂等操作——表已存在就什么都不做，
    所以程序每次启动调用它都很安全。

    外键关系（REFERENCES）：
      chunks.doc_id            -> documents.id        （一篇文档切成多个切片）
      messages.conversation_id -> conversations.id    （一个会话有多条消息）
    """
    with get_conn() as conn:
        # with conn 会自动处理事务：正常结束就提交（commit），出错就回滚（rollback）。
        # executescript 可以一次性执行多条 SQL 语句（以分号分隔）。
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键，每条记录的身份证
                title TEXT NOT NULL,                   -- 文档标题
                source TEXT NOT NULL DEFAULT 'unknown',-- 来源（文件名/数据集名）
                content TEXT NOT NULL,                 -- 文档全文
                created_at TEXT NOT NULL               -- 入库时间
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL REFERENCES documents(id), -- 属于哪篇文档
                chunk_index INTEGER NOT NULL,          -- 这是该文档的第几段（从 0 开始）
                content TEXT NOT NULL,                 -- 切片内容
                token_count INTEGER NOT NULL DEFAULT 0 -- 估算的 token 数（备用字段）
            );
            -- 索引：按 doc_id 查切片时会更快。数据量小时感觉不到差别。
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL,        -- 'user'（用户）或 'assistant'（助手）
                content TEXT NOT NULL,     -- 消息内容
                created_at TEXT NOT NULL
            );
            """
        )


def now() -> str:
    """返回当前时间，格式为 ISO 8601 字符串（例如 2026-08-07T15:30:00）。"""
    return datetime.now().isoformat(timespec="seconds")


# ---- documents / chunks ----

def doc_exists(title: str, source: str) -> bool:
    """判断某篇文档是否已导入过（按 title + source 唯一识别）。

    用于去重：导入时发现已存在就跳过，避免重复入库。
    SQL 里的问号 ? 是参数化查询占位符，由 SQLite 负责转义，
    这是防止 SQL 注入的标准做法，比把变量拼进 SQL 字符串安全。
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE title = ? AND source = ?",
            (title, source),
        ).fetchone()
    return row is not None  # 查到任意一行就说明已存在


def insert_document(title: str, source: str, content: str) -> int:
    """插入一篇文档，返回它的自增主键 id。

    cur.lastrowid 是 SQLite 给这条新记录分配的自增 id。
    ingest.py 会拿它当 chunks.doc_id，把切片挂到这篇文档下面。
    """
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (title, source, content, created_at) VALUES (?, ?, ?, ?)",
            (title, source, content, now()),
        )
        return cur.lastrowid


def insert_chunk(doc_id: int, chunk_index: int, content: str) -> int:
    """插入一个切片，返回切片 id。

    token_count 用字符数/2 粗略估算（中文约 2 个字符 = 1 个 token）。
    只是估算值，目前没有被检索逻辑用到，属于预留字段。
    """
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, content, token_count) VALUES (?, ?, ?, ?)",
            (doc_id, chunk_index, content, max(1, len(content) // 2)),
        )
        return cur.lastrowid


# ---- conversations / messages ----

def create_conversation() -> int:
    """新建一次对话，返回对话 id（前端第一次提问时后端会调用它）。"""
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO conversations (created_at) VALUES (?)", (now(),))
        return cur.lastrowid


def add_message(conversation_id: int, role: str, content: str) -> int:
    """往某个会话里追加一条消息（用户问题或助手回答）。"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now()),
        )
        return cur.lastrowid


def get_messages(conversation_id: int, limit: int = 20) -> list[dict]:
    """取出某个会话最近 limit 条消息，按时间从旧到新返回。

    SQL 先按 id 倒序（DESC）取最新的 limit 条，
    再用 list(reversed(...)) 翻回正序，方便直接丢给大模型当上下文。
    查询结果是 sqlite3.Row 对象，先 dict() 转成普通字典再返回。
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    return list(reversed([dict(r) for r in rows]))


def stats() -> dict:
    """统计各表有多少条记录，供 /health 接口查看系统状态。"""
    with get_conn() as conn:
        # fetchone()["n"] 取 COUNT(*) 的别名 n（AS n 就是给结果列起名）。
        docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        msgs = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
    return {"documents": docs, "chunks": chunks, "messages": msgs}
