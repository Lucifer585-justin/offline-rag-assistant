"""
rag_engine.py — 检索增强引擎（文档 → 向量库 → 检索）

把原来拆在两个文件里的 build_db.py（建库）和 retriever.py（检索）合并成一个类，
因为它们的生命周期是绑定的：建库后要立刻让检索看到新数据，分两个模块就得靠调用方
自己协调"重新加载"，很容易出 bug（改了库但内存里还是旧矩阵）。

数据流：
    PDF/TXT/MD ──抽取──▶ 原文 ──清洗──▶ 段落 ──切块(500/100)──▶ chunks
                                        └── bge-m3 编码 ──▶ float32 向量
                                                              │
                                                              ▼
                                                    SQLite (chunks 表)
                                                              │
    用户问题 ──bge-m3 编码──▶ q ── 余弦相似度(矩阵乘法) ──▶ top-k 片段

为什么用 SQLite + numpy 而不是 Chroma/FAISS：
    · 零额外依赖：sqlite3 是标准库，numpy 只做一次矩阵乘法；
    · 完全透明：相似度就是 dot(q, M)，没有黑盒索引，讲得清；
    · 规模匹配：1 万块以内暴力检索 5~20ms，比服务化方案更快；
    · 单文件：knowledge.db 拷走就能用。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from config import Config
from logger import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    page   INTEGER NOT NULL DEFAULT 0,
    seq    INTEGER NOT NULL DEFAULT 0,
    text   TEXT NOT NULL,
    vec    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunks(source);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

SUPPORTED_EXT = {".pdf", ".txt", ".md", ".markdown"}


class RAGError(RuntimeError):
    """检索/入库相关的业务异常，交给上层转成 4xx/5xx。"""


# ==============================================================================
# 文本处理（纯函数，方便单测）
# ==============================================================================
def clean_text(text: str) -> str:
    """
    清洗 PDF 抽出来的脏文本，三类噪声最常见：
      1. 纯页码行（"12" / "- 12 -" / "第 12 页"）
      2. 中文行尾被硬换行切断（PDF 按版面断行，不是按语义）
      3. 连续空行 / 多余空格
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"(?m)^\s*(?:[-—–]\s*)?\d{1,4}\s*(?:[-—–])?\s*$", "", text)
    text = re.sub(r"(?m)^\s*第\s*\d+\s*页\s*$", "", text)
    # 上一行不是句末标点、下一行以中文开头 → 合并（修复 PDF 硬换行）
    text = re.sub(r"(?<=[^\n。！？；：\.\!\?;:])\n(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """
    切块策略：按段落累积到 size 就封口，并从块尾取 overlap 字作为下一块开头。
    重叠的作用：答案正好跨在切口上时，至少有一块是完整的。

    超长段落（比如整章没有空行）会先按句号拆成句子，再逐句塞。
    """
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = buf[-overlap:] if overlap > 0 else ""

    for p in paras:
        if len(p) > size:
            for sent in re.split(r"(?<=[。！？；\.\!\?;])", p):
                if not sent.strip():
                    continue
                if len(buf) + len(sent) > size:
                    flush()
                buf += sent
        else:
            if len(buf) + len(p) + 1 > size:
                flush()
            buf += p + "\n"
    if buf.strip():
        chunks.append(buf.strip())

    return [c for c in chunks if len(c) >= 20]


# ==============================================================================
# 引擎
# ==============================================================================
class RAGEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db_path = str(cfg.kb_path)
        self._lock = threading.RLock()          # 上传与检索并发时保护内存矩阵
        self._matrix: np.ndarray | None = None  # (n, dim) 已归一化
        self._rows: list[dict] = []             # 与矩阵行一一对应
        self._dim = 0
        self._model = cfg.embed_model
        self._error: str | None = None

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)
        self.reload()

    # ------------------------------------------------------------------ 基础
    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=15, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ------------------------------------------------------------------ 载入
    def reload(self) -> None:
        """把全库向量读进内存并预先归一化（检索时点积即余弦相似度）。"""
        with self._lock:
            try:
                with self._conn() as con:
                    rows = con.execute(
                        "SELECT id, source, page, seq, text, vec FROM chunks ORDER BY id"
                    ).fetchall()
                    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())

                if not rows:
                    self._matrix, self._rows, self._dim = None, [], 0
                    self._error = "知识库为空，请先上传文档"
                    return

                vecs, metas = [], []
                for r in rows:
                    vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
                    metas.append({"id": r["id"], "source": r["source"],
                                  "page": r["page"], "seq": r["seq"], "text": r["text"]})

                matrix = np.vstack(vecs).astype(np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._matrix = matrix / norms
                self._rows = metas
                self._dim = matrix.shape[1]

                # 维度校验：换过 embedding 模型却没重建库，检索结果会完全错乱
                saved_dim = meta.get("dim")
                if saved_dim and int(saved_dim) != self._dim:
                    self._error = f"向量维度不一致（库 {saved_dim} / 当前 {self._dim}），请重建知识库"
                    log.error(self._error)
                else:
                    self._error = None
                log.info("知识库载入：%d 块 / %d 维", len(self._rows), self._dim)
            except Exception as exc:  # noqa: BLE001
                self._error = f"载入知识库失败：{exc}"
                log.exception("载入知识库失败")

    # ------------------------------------------------------------------ 编码
    def embed(self, texts: list[str], timeout: int = 300) -> np.ndarray:
        """
        调 Ollama /api/embed 批量编码。
        options.num_gpu=0：把 embedding 钉在 CPU —— 8GB 显存里 bge-m3 和 7B 抢显存时，
        Ollama 会把整个 7B 退回 CPU，生成速度从 58 tok/s 掉到 5 tok/s。
        bge-m3 在 CPU 上编码一句话约 100ms，宁可慢一点也要把 GPU 让给生成模型。
        """
        payload = json.dumps({
            "model": self.cfg.embed_model,
            "input": texts,
            "options": {"num_gpu": 0},
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.ollama_base}/api/embed", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RAGError(f"embedding 接口返回 {exc.code}，检查模型 {self.cfg.embed_model} 是否存在") from exc
        except urllib.error.URLError as exc:
            raise RAGError(f"连不上 Ollama（{self.cfg.ollama_base}）：{exc.reason}") from exc

        if "embeddings" not in data:
            raise RAGError(f"embedding 响应异常：{str(data)[:200]}")
        return np.asarray(data["embeddings"], dtype=np.float32)

    # ------------------------------------------------------------------ 抽取
    def extract_text(self, path: Path) -> list[tuple[int, str]]:
        """返回 [(页码, 文本)]。扫描件 PDF 抽不出字，返回空页。"""
        ext = path.suffix.lower()
        if ext == ".pdf":
            try:
                try:
                    import pymupdf as fitz          # 新版包名
                except ImportError:
                    import fitz                     # 旧版包名，仍兼容
                with fitz.open(path) as doc:
                    return [(i, page.get_text("text")) for i, page in enumerate(doc, start=1)]
            except ImportError:
                from pypdf import PdfReader
                return [(i, page.extract_text() or "")
                        for i, page in enumerate(PdfReader(str(path)).pages, start=1)]
        if ext in {".txt", ".md", ".markdown"}:
            for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
                try:
                    return [(0, path.read_text(encoding=enc))]
                except UnicodeDecodeError:
                    continue
            return [(0, path.read_text(encoding="utf-8", errors="ignore"))]
        raise RAGError(f"不支持的文件类型：{ext}")

    # ------------------------------------------------------------------ 入库
    def add_document(self, path: Path) -> dict:
        """
        解析 → 清洗 → 切块 → 编码 → 写库 → 刷新内存矩阵。
        同名文档重复上传时先清旧块（相当于覆盖更新）。
        """
        started = time.time()
        source = path.name
        pages = self.extract_text(path)
        pieces: list[tuple[str, int]] = []
        for page_no, raw in pages:
            for ck in split_chunks(clean_text(raw), self.cfg.chunk_size, self.cfg.chunk_overlap):
                pieces.append((ck, page_no))

        if not pieces:
            raise RAGError("没有抽到可用文本（扫描版 PDF 需要先做 OCR）")

        vectors = []
        for i in range(0, len(pieces), self.cfg.embed_batch):
            batch = [p[0] for p in pieces[i:i + self.cfg.embed_batch]]
            vectors.append(self.embed(batch))
        matrix = np.vstack(vectors)

        with self._lock, self._conn() as con:
            con.execute("DELETE FROM chunks WHERE source=?", (source,))
            con.executemany(
                "INSERT INTO chunks(source, page, seq, text, vec) VALUES(?,?,?,?,?)",
                [(source, pieces[i][1], i, pieces[i][0], matrix[i].tobytes())
                 for i in range(len(pieces))],
            )
            con.execute("INSERT OR REPLACE INTO meta VALUES('embed_model', ?)", (self.cfg.embed_model,))
            con.execute("INSERT OR REPLACE INTO meta VALUES('dim', ?)", (str(matrix.shape[1]),))
            con.execute("INSERT OR REPLACE INTO meta VALUES('updated_at', ?)",
                        (time.strftime("%Y-%m-%d %H:%M:%S"),))
            con.commit()

        self.reload()
        elapsed = int((time.time() - started) * 1000)
        log.info("文档入库：%s → %d 块 / %d 页 / %d ms", source, len(pieces), len(pages), elapsed)
        return {"source": source, "pages": len(pages), "chunks": len(pieces), "ms": elapsed}

    def build_from_dir(self, reset: bool = False) -> dict:
        """扫描 data/docs 下所有文档并逐个入库（首次初始化 / 重建用）。"""
        if reset:
            with self._lock, self._conn() as con:
                con.execute("DELETE FROM chunks")
                con.commit()
            log.warning("知识库已清空（reset=True）")

        results, errors = [], []
        for path in sorted(Path(self.cfg.docs_dir).rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
                try:
                    results.append(self.add_document(path))
                except Exception as exc:  # noqa: BLE001
                    log.exception("入库失败：%s", path.name)
                    errors.append({"source": path.name, "error": str(exc)})
        return {"ok": len(errors) == 0, "documents": results, "errors": errors,
                "chunks": sum(r["chunks"] for r in results)}

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int | None = None,
               min_score: float | None = None) -> tuple[list[dict], float]:
        """
        返回 (hits, best_score)。hits 为空表示"没找到相关内容"，调用方应据此拒答。
        相似度 = 归一化后的点积，一次矩阵乘法算完全库。
        """
        top_k = top_k or self.cfg.top_k
        thr = self.cfg.min_score if min_score is None else min_score

        with self._lock:
            if self._matrix is None or not self._rows:
                return [], 0.0
            matrix, rows = self._matrix, self._rows

        q = self.embed([query])[0]
        if q.shape[0] != matrix.shape[1]:
            raise RAGError(f"问题向量维度 {q.shape[0]} 与库 {matrix.shape[1]} 不一致，请重建知识库")
        q = q / (np.linalg.norm(q) or 1.0)

        scores = matrix @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        hits = []
        for i in idx:
            s = float(scores[i])
            if s < thr:
                continue
            row = dict(rows[int(i)])
            row["score"] = round(s, 4)
            hits.append(row)
        best = float(scores[idx[0]]) if len(idx) else 0.0

        # ------------------------------------------------------------------
        # 区分度判据（margin）：只看绝对分数会误召回。
        # 实测反例：问"写一个 python 快速排序"，计组教材的 3 个片段拿到
        # 0.4605 / 0.4557 / 0.4539 —— 全部擦着阈值过线，且彼此只差 0.0066。
        # 这说明模型对这段文本没有"明显偏好"，只是泛泛相关（背景噪声）。
        # 真命中时通常有明显断层：0.6275 / 0.6134 / 0.6059（差 0.0216）。
        # 所以：top1 - topN < margin 时判为噪声，直接当作没找到。
        # ------------------------------------------------------------------
        if len(hits) >= 2:
            margin = hits[0]["score"] - hits[-1]["score"]
            if margin < self.cfg.min_margin and hits[0]["score"] < 0.62:
                log.info("检索区分度不足，判为噪声：margin=%.4f (<%.3f) best=%.3f，已丢弃 %d 段",
                         margin, self.cfg.min_margin, hits[0]["score"], len(hits))
                return [], best

        return hits, best

    # ------------------------------------------------------------------ 管理
    @property
    def ready(self) -> bool:
        """知识库是否可用（有向量且维度没出错）。空库时上层应退化为普通聊天。"""
        return self._matrix is not None and not self._error

    def list_documents(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT source, COUNT(*) AS chunks, SUM(LENGTH(text)) AS chars, "
                "MIN(page) AS first_page, MAX(page) AS last_page "
                "FROM chunks GROUP BY source ORDER BY source"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, source: str) -> int:
        with self._lock, self._conn() as con:
            cur = con.execute("DELETE FROM chunks WHERE source=?", (source,))
            con.commit()
        # 同时删掉磁盘上的原文件，避免"删了但重启又被 build_from_dir 拉回来"
        for p in Path(self.cfg.docs_dir).rglob(source):
            try:
                p.unlink()
            except OSError:
                pass
        self.reload()
        log.info("删除文档：%s（%d 块）", source, cur.rowcount)
        return cur.rowcount

    def stats(self) -> dict:
        docs = self.list_documents()
        return {
            "ready": self._matrix is not None,
            "chunks": len(self._rows),
            "documents": len(docs),
            "dim": self._dim,
            "embed_model": self._model,
            "min_score": self.cfg.min_score,
            "error": self._error,
            "items": docs,
        }
