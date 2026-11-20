"""
memory.py — 会话记忆（SQLite）

职责边界：**只负责存取，不负责裁剪**。
    这里只管"谁在什么时候说了什么"；
    "最多带几轮进 prompt" 是 rag_engine / app 层的上下文预算问题。
    分开的好处：换存储（SQLite → Redis）不影响业务逻辑，换裁剪策略也不动表结构。

表设计：
    sessions(sid PK, created_at, updated_at, turns)
    messages(id PK, sid, role, content, created_at)   INDEX(sid, id)

并发：
    每次操作新建连接、用完即关。SQLite 打开开销极小（微秒级），
    换来的是线程安全 —— Flask threaded=True 时不存在共享连接被多线程踩的问题。
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    sid        TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    turns      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sid        TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_sid ON messages(sid, id);
"""


class SessionStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)
        log.info("会话存储就绪 → %s", self.db_path)

    # ------------------------------------------------------------------ 基础
    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        con.row_factory = sqlite3.Row
        # WAL 模式：读写不互相阻塞，多个请求同时进来时体验更好
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ------------------------------------------------------------------ 会话
    @staticmethod
    def new_sid() -> str:
        return uuid.uuid4().hex

    def touch(self, sid: str, turns_delta: int = 0) -> None:
        now = time.time()
        with self._conn() as con:
            con.execute(
                "INSERT INTO sessions(sid, created_at, updated_at, turns) VALUES(?,?,?,?) "
                "ON CONFLICT(sid) DO UPDATE SET updated_at=excluded.updated_at, "
                "turns = sessions.turns + ?",
                (sid, now, now, max(0, turns_delta), max(0, turns_delta)),
            )

    def exists(self, sid: str) -> bool:
        with self._conn() as con:
            return con.execute("SELECT 1 FROM sessions WHERE sid=?", (sid,)).fetchone() is not None

    # ------------------------------------------------------------------ 消息
    def add(self, sid: str, role: str, content: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO messages(sid, role, content, created_at) VALUES(?,?,?,?)",
                (sid, role, content, time.time()),
            )

    def add_turn(self, sid: str, question: str, answer: str) -> None:
        """一问一答一次写入，减少事务次数。"""
        now = time.time()
        with self._conn() as con:
            con.executemany(
                "INSERT INTO messages(sid, role, content, created_at) VALUES(?,?,?,?)",
                [(sid, "user", question, now), (sid, "assistant", answer, now)],
            )

    def history(self, sid: str, limit: int = 40) -> list[dict]:
        """
        取最近 limit 条，按时间正序返回。
        先倒序 LIMIT（走索引尾部），再反转 —— 比全表 ORDER BY 快得多。
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT role, content FROM messages WHERE sid=? ORDER BY id DESC LIMIT ?",
                (sid, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def count(self, sid: str) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM messages WHERE sid=?", (sid,)).fetchone()[0]

    def clear(self, sid: str) -> int:
        """清空记忆：删消息 + 轮数归零，会话本身保留（Cookie 不变，用户无感）。"""
        with self._conn() as con:
            cur = con.execute("DELETE FROM messages WHERE sid=?", (sid,))
            con.execute("UPDATE sessions SET turns=0, updated_at=? WHERE sid=?", (time.time(), sid))
        log.info("会话 %s… 记忆已清空（删除 %d 条）", sid[:8], cur.rowcount)
        return cur.rowcount

    # ------------------------------------------------------------------ 维护
    def prune(self, keep_days: int = 30) -> int:
        cutoff = time.time() - keep_days * 86400
        with self._conn() as con:
            dead = [r["sid"] for r in con.execute(
                "SELECT sid FROM sessions WHERE updated_at < ?", (cutoff,)).fetchall()]
            for sid in dead:
                con.execute("DELETE FROM messages WHERE sid=?", (sid,))
                con.execute("DELETE FROM sessions WHERE sid=?", (sid,))
        if dead:
            log.info("清理了 %d 个过期会话", len(dead))
        return len(dead)
