"""
config.py — 全局配置

设计原则：
  1. **一切可变项走环境变量**，代码里不留硬编码路径；
  2. 用 frozen dataclass，避免运行期被误改；
  3. 提供 `Config.load()` 单入口，测试时可以注入自定义配置。

环境变量一览（也可写进 .env，本项目不引入 python-dotenv，保持零依赖）：
    OLLAMA_BASE          Ollama 地址       默认 http://127.0.0.1:11434
    OLLAMA_MODEL         生成模型          默认 qwen2.5-coder:7b
    OLLAMA_EMBED_MODEL   向量模型          默认 bge-m3
    OLLAMA_NUM_CTX       上下文窗口        默认 8192
    RESERVE_OUTPUT       输出预留 token    默认 1200
    RAG_TOP_K            检索条数          默认 3
    RAG_MIN_SCORE        相似度阈值        默认 0.45
    CHUNK_SIZE           切块字数          默认 500
    CHUNK_OVERLAP        重叠字数          默认 100
    WEB_HOST/WEB_PORT    Flask 绑定        默认 0.0.0.0:5000
    MAX_UPLOAD_MB        上传大小上限      默认 50
    LOG_LEVEL            日志级别          默认 INFO
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # ---- 路径 ----
    base_dir: Path
    data_dir: Path
    docs_dir: Path
    db_dir: Path
    log_dir: Path
    kb_path: Path
    sessions_path: Path

    # ---- 模型 ----
    ollama_base: str
    model: str
    embed_model: str
    num_ctx: int
    reserve_output: int

    # ---- 检索 ----
    top_k: int
    min_score: float
    chunk_size: int
    chunk_overlap: int
    embed_batch: int

    # ---- Web ----
    host: str
    port: int
    max_upload_mb: int

    # ---- 日志 ----
    log_level: str

    @property
    def prompt_budget(self) -> int:
        """prompt 可用的 token 上限 = 上下文窗口 - 输出预留。"""
        return max(512, self.num_ctx - self.reserve_output)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def load(cls, base_dir: Path | None = None) -> "Config":
        root = Path(base_dir) if base_dir else BASE_DIR
        data = root / "data"
        cfg = cls(
            base_dir=root,
            data_dir=data,
            docs_dir=data / "docs",
            db_dir=data / "db",
            log_dir=root / "logs",
            kb_path=data / "db" / "knowledge.db",
            sessions_path=data / "db" / "sessions.db",
            ollama_base=_env_str("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/"),
            model=_env_str("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            embed_model=_env_str("OLLAMA_EMBED_MODEL", "bge-m3"),
            num_ctx=_env_int("OLLAMA_NUM_CTX", 8192),
            reserve_output=_env_int("RESERVE_OUTPUT", 1200),
            top_k=_env_int("RAG_TOP_K", 3),
            min_score=_env_float("RAG_MIN_SCORE", 0.45),
            chunk_size=_env_int("CHUNK_SIZE", 500),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
            embed_batch=_env_int("EMBED_BATCH", 16),
            host=_env_str("WEB_HOST", "0.0.0.0"),
            port=_env_int("WEB_PORT", 5000),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        )
        for d in (cfg.data_dir, cfg.docs_dir, cfg.db_dir, cfg.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        return cfg

    def summary(self) -> dict:
        """启动时打印 / 写日志用，便于排查"为什么模型不在 D 盘"这类问题。"""
        return {
            "model": self.model,
            "embed_model": self.embed_model,
            "ollama_base": self.ollama_base,
            "num_ctx": self.num_ctx,
            "prompt_budget": self.prompt_budget,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "docs_dir": str(self.docs_dir),
            "kb_path": str(self.kb_path),
            "listen": f"{self.host}:{self.port}",
        }
