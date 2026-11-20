"""
logger.py — 统一日志

为什么不用 print：
  1. print 没有级别，线上出问题无法过滤；
  2. print 的输出去不了文件，重启后日志就没了；
  3. logging 自带轮转（RotatingFileHandler），不会把磁盘写满。

用法：
    from logger import setup_logging, get_logger
    setup_logging(cfg.log_dir, cfg.log_level)      # 进程启动时调一次
    log = get_logger(__name__)
    log.info("...")
    log.exception("失败了")                        # 自动带堆栈
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False

# 统一格式：时间 | 级别 | 模块 | 消息
FMT = "%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path | str, level: str = "INFO") -> None:
    """
    初始化根 logger。重复调用只生效一次（Flask 的 reloader 会 import 两次）。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 控制台：开发时直接看
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(FMT, DATEFMT))
    root.addHandler(console)

    # 文件：单文件 5MB，保留 3 份，够排查又不撑爆磁盘
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=5 * 1024 * 1024,
        backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(FMT, DATEFMT))
    root.addHandler(file_handler)

    # 降低第三方库噪音：werkzeug 的每条请求日志在演示时太吵
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger("bootstrap").info("日志系统就绪 → %s", log_dir / "app.log")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
