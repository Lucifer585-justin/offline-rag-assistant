#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_rag.py — 检索层评测

为什么要有评测集：
    调阈值、换 embedding、加 BM25，每次都说"感觉变好了"是不算数的。
    有了一组固定题目 + 固定判据，才能回答"到底好了多少、有没有变坏"。

三类指标：
    1. 召回命中率  —— expect=hit 的题，检索返回非空的比例
    2. 拒答准确率  —— expect=miss 的题，检索返回空的比例（防幻觉的核心指标）
    3. 要点覆盖率  —— top-1 片段里是否真的包含答案关键词（粗粒度"检索到的是不是正确内容"）

用法：
    python scripts/eval_rag.py                # 只测检索层，几秒钟跑完
    python scripts/eval_rag.py --llm           # 额外跑端到端，检查回答是否含要点（慢）
    python scripts/eval_rag.py --json out.json # 导出机器可读结果，便于比对版本
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config                      # noqa: E402
from rag_engine import RAGEngine               # noqa: E402


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["items"] if isinstance(data, dict) else data


def keyword_coverage(text: str, keywords: list[str]) -> float:
    """top-1 片段覆盖了多少关键词。没有关键词的题（miss 类）不参与统计。"""
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k.lower() in text.lower())
    return hit / len(keywords)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data" / "eval" / "questions.json"))
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--llm", action="store_true", help="额外跑端到端回答（慢）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    cfg = Config.load()
    engine = RAGEngine(cfg)
    if not engine.ready:
        print(f"[×] 知识库不可用：{engine.stats()['error']}")
        return 1

    items = load_questions(Path(args.questions))
    print(f"知识库：{len(engine._rows)} 块 / {engine._dim} 维 / 阈值 {cfg.min_score} / margin {cfg.min_margin}")
    print(f"评测题：{len(items)} 道\n")

    hit_total = hit_ok = 0
    miss_total = miss_ok = 0
    covers, latencies, failures = [], [], []

    for it in items:
        q, expect = it["q"], it["expect"]
        t0 = time.perf_counter()
        hits, best = engine.search(q)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        if expect == "hit":
            hit_total += 1
            if hits:
                hit_ok += 1
                cov = keyword_coverage(hits[0]["text"], it.get("keywords", []))
                covers.append(cov)
                if cov < 0.5:
                    failures.append({"q": q, "type": "低覆盖", "cov": round(cov, 2),
                                     "top1": hits[0]["text"][:60], "score": hits[0]["score"]})
            else:
                failures.append({"q": q, "type": "误拒", "best": round(best, 3)})
        else:
            miss_total += 1
            if not hits:
                miss_ok += 1
            else:
                failures.append({"q": q, "type": "误召回", "score": hits[0]["score"],
                                 "top1": hits[0]["text"][:60]})

    recall = hit_ok / hit_total * 100 if hit_total else 0
    reject = miss_ok / miss_total * 100 if miss_total else 0
    avg_cov = statistics.mean(covers) * 100 if covers else 0
    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1] if latencies else 0

    print(f"召回命中率   {hit_ok}/{hit_total}  = {recall:.1f}%")
    print(f"拒答准确率   {miss_ok}/{miss_total} = {reject:.1f}%")
    print(f"要点覆盖率   {avg_cov:.1f}%   (top-1 片段包含答案关键词的比例)")
    print(f"检索延迟     p50={p50:.0f} ms  p95={p95:.0f} ms")

    if failures:
        print(f"\n失败用例 {len(failures)} 条：")
        for f in failures[:12]:
            print(f"  [{f['type']}] {f['q']}  {({k: v for k, v in f.items() if k not in ('q', 'type')})}")

    if args.llm:
        print("\n== 端到端（调用 /api/chat，每题约 2~5s）==")
        import urllib.request
        ok = 0
        for it in items:
            if it["expect"] != "hit":
                continue
            body = json.dumps({"message": it["q"]}, ensure_ascii=False).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{cfg.port}/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                text = r.read().decode()
            cov = keyword_coverage(text, it.get("keywords", []))
            ok += 1 if cov >= 0.5 else 0
            print(f"  {'✓' if cov >= 0.5 else '×'} {it['q']}  覆盖 {cov:.0%}")
        print(f"端到端要点覆盖：{ok}/{hit_total}")

    result = {
        "knowledge_base": {"chunks": len(engine._rows), "dim": engine._dim,
                           "min_score": cfg.min_score, "min_margin": cfg.min_margin,
                           "embed_model": cfg.embed_model},
        "recall_at_k": round(recall, 2),
        "reject_accuracy": round(reject, 2),
        "keyword_coverage": round(avg_cov, 2),
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1)},
        "failures": failures,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")

    # 退化阈值：拒答低于 90% 说明防幻觉开始漏，召回低于 80% 说明检索不达标
    return 0 if (recall >= 80 and reject >= 90) else 1


if __name__ == "__main__":
    sys.exit(main())
