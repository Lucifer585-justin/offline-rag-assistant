"""
rag_engine.py — 检索增强引擎（混合检索版）

相对上一版的三个升级：
  1. **混合检索**：向量（语义）+ BM25（关键词）双路召回，RRF 融合排名。
     解决的痛点：问"组相联映射的 k 是多少"时，纯向量会飘到讲"Cache 结构"的泛段落，
     而真正答案里恰好有"组相联"三个字 —— BM25 一抓就中。
  2. **引用可定位**：每个 chunk 记录它在所属页文本里的字符区间 (char_start, char_end)，
     前端点 [1] 就能展开原文并高亮那一段。
  3. **增量索引**：files 表记录 sha256 + mtime，重建时跳过没变的文件。

数据模型：
    chunks(id, source, page, seq, text, vec, char_start, char_end)
    pages (source, page, text)          ← 存清洗后的页文本，供前端展示原文
    files (source, sha256, size, mtime, chunks, indexed_at)   ← 增量索引用
    meta  (key, value)
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from bm25 import BM25, rrf_fuse
from config import Config
from logger import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    page       INTEGER NOT NULL DEFAULT 0,
    seq        INTEGER NOT NULL DEFAULT 0,
    text       TEXT NOT NULL,
    vec        BLOB NOT NULL,
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunks(source);
CREATE TABLE IF NOT EXISTS pages (
    source TEXT NOT NULL,
    page   INTEGER NOT NULL,
    text   TEXT NOT NULL,
    PRIMARY KEY (source, page)
);
CREATE TABLE IF NOT EXISTS files (
    source     TEXT PRIMARY KEY,
    sha256     TEXT,
    size       INTEGER,
    mtime      REAL,
    chunks     INTEGER,
    indexed_at REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

SUPPORTED_EXT = {".pdf", ".txt", ".md", ".markdown"}


class RAGError(RuntimeError):
    """检索/入库相关的业务异常，交给上层转成 4xx/5xx。"""


# ==============================================================================
# 文本处理
# ==============================================================================
def clean_text(text: str) -> str:
    """清洗 PDF 文本：去页码行、修复硬换行、压掉多余空行。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"(?m)^\s*(?:[-—–]\s*)?\d{1,4}\s*(?:[-—–])?\s*$", "", text)
    text = re.sub(r"(?m)^\s*第\s*\d+\s*页\s*$", "", text)
    text = re.sub(r"(?<=[^\n。！？；：\.\!\?;:])\n(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_chunks_spans(text: str, size: int = 500, overlap: int = 100):
    """
    切块并返回每块在原文中的字符区间：[(chunk_text, start, end), ...]

    实现要点：**不做字符串拼接**，只用"区间列表 + 原文切片"。
    上一版用 buf += s 累加，再用 len(buf) 反推位置，一旦重叠或多段拼接就会漂移
    1~2 个字符（实测 7 块里错 3 块），前端高亮就会圈错字。
    现在所有 offset 直接来自 re.finditer 的 match.start()/end()，绝对对齐。

    切块单位是"句子"（按中文标点/换行切），再按 size 累积：
      - 句子边界天然适合作为语义边界
      - 单段超过 size 时，句子本身会被拆开累积，不会撑爆
    """
    if not text:
        return []

    units = [(m.start(), m.end()) for m in
             re.finditer(r"[^。！？；\.\!\?;\n]+[。！？；\.\!\?;]?", text)
             if m.group().strip()]
    if not units:
        return []

    out = []
    i = 0
    n = len(units)
    while i < n:
        start = units[i][0]
        end = units[i][1]
        j = i
        # 向后吞句子，直到再加一句就会超过 size
        while j + 1 < n and units[j + 1][1] - start <= size:
            j += 1
            end = units[j][1]

        chunk = text[start:end].strip()
        if len(chunk) >= 20:
            out.append((chunk, start, end))

        if j >= n - 1:
            break

        # 重叠：下一块的起点回退 overlap 字符，再对齐到最近的句子起点
        target = max(start + 1, end - overlap)
        k = j + 1
        while k < n and units[k][0] < target:
            k += 1
        i = k if k > j else j + 1

    return out


# ==============================================================================
# 引擎
# ==============================================================================
class RAGEngine:
    def __init__(self, cfg: Config, db_path: Path | None = None):
        self.cfg = cfg
        self.db_path = str(db_path or cfg.kb_path)
        self._lock = threading.RLock()
        self._matrix: np.ndarray | None = None
        self._rows: list[dict] = []
        self._bm25: BM25 | None = None
        self._dim = 0
        self._error: str | None = None

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)
            self._migrate(con)
        self.reload()

    @staticmethod
    def _migrate(con: sqlite3.Connection) -> None:
        """老库没有 char_start/char_end 列，就地补上，避免用户必须重建。"""
        cols = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
        if "char_start" not in cols:
            con.execute("ALTER TABLE chunks ADD COLUMN char_start INTEGER NOT NULL DEFAULT 0")
        if "char_end" not in cols:
            con.execute("ALTER TABLE chunks ADD COLUMN char_end INTEGER NOT NULL DEFAULT 0")
        con.commit()

    # ------------------------------------------------------------------ 基础
    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=15, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ------------------------------------------------------------------ 载入
    def reload(self) -> None:
        """载入向量矩阵 + 构建 BM25 索引（两者都常驻内存）。"""
        with self._lock:
            try:
                with self._conn() as con:
                    rows = con.execute(
                        "SELECT id, source, page, seq, text, vec, char_start, char_end "
                        "FROM chunks ORDER BY id").fetchall()
                    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())

                if not rows:
                    self._matrix, self._rows, self._bm25, self._dim = None, [], None, 0
                    self._error = "知识库为空，请先上传文档"
                    return

                vecs, metas, texts = [], [], []
                for r in rows:
                    vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
                    metas.append({
                        "id": r["id"], "source": r["source"], "page": r["page"],
                        "seq": r["seq"], "text": r["text"],
                        "char_start": r["char_start"], "char_end": r["char_end"],
                    })
                    texts.append(r["text"])

                matrix = np.vstack(vecs).astype(np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._matrix = matrix / norms
                self._rows = metas
                self._dim = matrix.shape[1]

                self._bm25 = BM25()
                self._bm25.fit(texts)

                saved_dim = meta.get("dim")
                if saved_dim and int(saved_dim) != self._dim:
                    self._error = f"向量维度不一致（库 {saved_dim} / 当前 {self._dim}），请重建知识库"
                    log.error(self._error)
                else:
                    self._error = None
                log.info("知识库载入：%d 块 / %d 维 / BM25 词表 %d",
                         len(self._rows), self._dim, len(self._bm25.df))
            except Exception as exc:  # noqa: BLE001
                self._error = f"载入知识库失败：{exc}"
                log.exception("载入知识库失败")

    # ------------------------------------------------------------------ 编码
    def embed(self, texts: list[str], timeout: int = 300) -> np.ndarray:
        """批量编码；num_gpu=0 把 embedding 钉在 CPU，不跟生成模型抢显存。"""
        payload = json.dumps({
            "model": self.cfg.embed_model,
            "input": texts,
            "options": {"num_gpu": 0},
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.ollama_base}/api/embed", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
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
        ext = path.suffix.lower()
        if ext == ".pdf":
            try:
                try:
                    import pymupdf as fitz
                except ImportError:
                    import fitz
                with fitz.open(path) as doc:
                    return [(i, page.get_text("text")) for i, page in enumerate(doc, start=1)]
            except ImportError:
                from pypdf import PdfReader
                return [(i, page.get_text() or "")
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
    def add_document(self, path: Path, force: bool = False) -> dict:
        """
        解析 → 清洗 → 切块（带字符区间）→ 编码 → 写库 → 重建内存索引。
        增量：同一个文件 sha256 没变就跳过（force=True 强制重建）。
        """
        started = time.time()
        source = path.name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()

        if not force:
            with self._conn() as con:
                row = con.execute("SELECT sha256 FROM files WHERE source=?", (source,)).fetchone()
            if row and row["sha256"] == digest:
                log.info("跳过未变更文档：%s", source)
                return {"source": source, "skipped": True, "chunks": 0, "pages": 0, "ms": 0}

        pages = self.extract_text(path)
        page_rows, piece_rows, vectors = [], [], []

        for page_no, raw_text in pages:
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue
            page_rows.append((source, page_no, cleaned))
            for txt, s, e in split_chunks_spans(cleaned, self.cfg.chunk_size, self.cfg.chunk_overlap):
                piece_rows.append((page_no, len(piece_rows), txt, s, e))

        if not piece_rows:
            raise RAGError("没有抽到可用文本（扫描版 PDF 需要先做 OCR）")

        for i in range(0, len(piece_rows), self.cfg.embed_batch):
            batch = [p[2] for p in piece_rows[i:i + self.cfg.embed_batch]]
            vectors.append(self.embed(batch))
        matrix = np.vstack(vectors)

        with self._lock, self._conn() as con:
            con.execute("DELETE FROM chunks WHERE source=?", (source,))
            con.execute("DELETE FROM pages WHERE source=?", (source,))
            con.executemany("INSERT OR REPLACE INTO pages(source, page, text) VALUES(?,?,?)", page_rows)
            con.executemany(
                "INSERT INTO chunks(source, page, seq, text, vec, char_start, char_end) "
                "VALUES(?,?,?,?,?,?,?)",
                [(source, p[0], p[1], p[2], matrix[i].tobytes(), p[3], p[4])
                 for i, p in enumerate(piece_rows)])
            con.execute(
                "INSERT OR REPLACE INTO files(source, sha256, size, mtime, chunks, indexed_at) "
                "VALUES(?,?,?,?,?,?)",
                (source, digest, len(raw), path.stat().st_mtime, len(piece_rows), time.time()))
            con.execute("INSERT OR REPLACE INTO meta VALUES('embed_model', ?)", (self.cfg.embed_model,))
            con.execute("INSERT OR REPLACE INTO meta VALUES('dim', ?)", (str(matrix.shape[1]),))
            con.execute("INSERT OR REPLACE INTO meta VALUES('updated_at', ?)",
                        (time.strftime("%Y-%m-%d %H:%M:%S"),))
            con.commit()

        self.reload()
        elapsed = int((time.time() - started) * 1000)
        log.info("文档入库：%s → %d 块 / %d 页 / %d ms", source, len(piece_rows), len(page_rows), elapsed)
        return {"source": source, "pages": len(page_rows), "chunks": len(piece_rows),
                "ms": elapsed, "skipped": False}

    def build_from_dir(self, reset: bool = False) -> dict:
        """扫描 docs 目录批量入库；reset=True 先清空。未变更文件自动跳过。"""
        if reset:
            with self._lock, self._conn() as con:
                con.execute("DELETE FROM chunks")
                con.execute("DELETE FROM pages")
                con.execute("DELETE FROM files")
                con.commit()
            log.warning("知识库已清空（reset=True）")

        results, errors, skipped = [], [], 0
        for path in sorted(Path(self.cfg.docs_dir).rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
                try:
                    info = self.add_document(path)
                    if info.get("skipped"):
                        skipped += 1
                    else:
                        results.append(info)
                except Exception as exc:  # noqa: BLE001
                    log.exception("入库失败：%s", path.name)
                    errors.append({"source": path.name, "error": str(exc)})
        return {"ok": len(errors) == 0, "documents": results, "errors": errors,
                "chunks": sum(r["chunks"] for r in results), "skipped": skipped}

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int | None = None,
               min_score: float | None = None) -> tuple[list[dict], float]:
        """
        混合检索：向量 Top-N + BM25 Top-N → RRF 融合 → 取 top_k。
        阈值与区分度判据仍基于**向量分数**（语义相关性更接近"能不能答"）。
        """
        top_k = top_k or self.cfg.top_k
        thr = self.cfg.min_score if min_score is None else min_score

        with self._lock:
            if self._matrix is None or not self._rows:
                return [], 0.0
            matrix, rows, bm25 = self._matrix, self._rows, self._bm25

        q = self.embed([query])[0]
        if q.shape[0] != matrix.shape[1]:
            raise RAGError(f"问题向量维度 {q.shape[0]} 与库 {matrix.shape[1]} 不一致，请重建知识库")
        q = q / (np.linalg.norm(q) or 1.0)

        vec_scores = matrix @ q
        bm_scores = np.asarray(bm25.scores(query), dtype=np.float32) if bm25 else np.zeros_like(vec_scores)
        bm_max = float(bm_scores.max()) if len(bm_scores) else 0.0

        # 两路各取 4 倍候选，融合后再收敛到 top_k
        cand = min(len(vec_scores), max(top_k * 4, 20))
        v_idx = np.argsort(-vec_scores)[:cand].tolist()
        b_idx = np.argsort(-bm_scores)[:cand].tolist()
        fused = rrf_fuse([v_idx, b_idx])

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]

        hits = []
        for i, fscore in ranked:
            vs = float(vec_scores[i])
            bs = float(bm_scores[i])
            # 双通道放行：语义过阈值 OR 关键词强命中。
            # 关键词判据必须同时满足"绝对分够高"和"是 BM25 里的头名"：
            #   绝对分（>=1.0）过滤掉"算法""方式"这类到处都有的常见词（IDF 低，得分 < 1）
            #   相对分（>=60% 最高分）过滤掉一群词都很弱命中的块
            # 典型受益者："BTB 是什么" 向量分只有 0.25，但原文明确写着"转移目标缓冲器 BTB"
            kw_strong = bs >= self.cfg.bm25_min and bs >= bm_max * 0.6
            if vs < thr and not kw_strong:
                continue
            row = dict(rows[int(i)])
            row["score"] = round(vs, 4)          # 展示仍用余弦，含义直观
            row["bm25"] = round(bs, 3)
            row["rrf"] = round(fscore, 5)
            row["kw"] = kw_strong
            hits.append(row)

        best = float(vec_scores.max()) if len(vec_scores) else 0.0
        if not hits:
            return [], best

        # 单 hit 时算不出区分度，改用"擦线 + 关键词也不强"作为噪声判据。
        # 反例：问"pandas 数据清洗"，只有一段擦到 0.4506，BM25 几乎为 0。
        if len(hits) == 1:
            h = hits[0]
            if h["score"] < thr + 0.05 and not h["kw"]:
                log.info("单段擦线且无关键词命中，判为噪声：score=%.3f bm25=%.2f",
                         h["score"], h["bm25"])
                return [], best
            return hits, best

        # 多 hit 时看区分度：分数扁平且关键词也不强 = 没有真正匹配。
        # 注意：只有"来自不同页/不同文档"的命中才参与判定 ——
        # 相邻块因为有 100 字重叠，分数天然接近，拿它算 margin 会误杀正常命中
        # （实测："写直达和写回的区别" 0.529 被判噪声，就是被同页重叠块拖累的）。
        pages_seen = {(h["source"], h["page"]) for h in hits}
        if len(pages_seen) >= 2:
            margin = hits[0]["score"] - hits[-1]["score"]
            if margin < self.cfg.min_margin and hits[0]["score"] < 0.62 and not hits[0]["kw"]:
                log.info("检索区分度不足，判为噪声：margin=%.4f best=%.3f", margin, hits[0]["score"])
                return [], best

        return hits, best

    # ------------------------------------------------------------------ 原文
    def page_text(self, source: str, page: int) -> str | None:
        """取某文档某页的清洗后原文，供前端做引用高亮。"""
        with self._conn() as con:
            row = con.execute("SELECT text FROM pages WHERE source=? AND page=?",
                              (source, page)).fetchone()
        return row["text"] if row else None

    # ------------------------------------------------------------------ 管理
    @property
    def ready(self) -> bool:
        return self._matrix is not None and not self._error

    def list_documents(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT c.source, COUNT(*) AS chunks, SUM(LENGTH(c.text)) AS chars, "
                "MIN(c.page) AS first_page, MAX(c.page) AS last_page, "
                "f.size AS bytes, f.indexed_at "
                "FROM chunks c LEFT JOIN files f ON f.source = c.source "
                "GROUP BY c.source ORDER BY c.source").fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, source: str) -> int:
        with self._lock, self._conn() as con:
            cur = con.execute("DELETE FROM chunks WHERE source=?", (source,))
            con.execute("DELETE FROM pages WHERE source=?", (source,))
            con.execute("DELETE FROM files WHERE source=?", (source,))
            con.commit()
        # 同步删磁盘文件，避免下次 --rebuild 又把它拉回来
        target = Path(self.cfg.docs_dir) / source
        if target.exists():
            try:
                target.unlink()
            except OSError:
                log.warning("删除磁盘文件失败：%s", target)
        self.reload()
        log.info("删除文档：%s（%d 块）", source, cur.rowcount)
        return cur.rowcount

    def stats(self) -> dict:
        docs = self.list_documents()
        return {
            "ready": self.ready,
            "chunks": len(self._rows),
            "documents": len(docs),
            "dim": self._dim,
            "embed_model": self.cfg.embed_model,
            "min_score": self.cfg.min_score,
            "min_margin": self.cfg.min_margin,
            "hybrid": True,
            "error": self._error,
            "items": docs,
        }
