"""在 509 块的教材语料上重新校准阈值。用法：python scripts/tune.py"""

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config          # noqa: E402
from rag_engine import RAGEngine   # noqa: E402


def evaluate(engine, items, cfg):
    hit_ok = hit_tot = miss_ok = miss_tot = 0
    for it in items:
        hits, _ = engine.search(it["q"])
        if it["expect"] == "hit":
            hit_tot += 1
            hit_ok += 1 if hits else 0
        else:
            miss_tot += 1
            miss_ok += 1 if not hits else 0
    return hit_ok / hit_tot * 100, miss_ok / miss_tot * 100


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    cfg = Config.load()
    engine = RAGEngine(cfg)
    items = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))["items"]
    print(f"语料 {len(engine._rows)} 块，题目 {len(items)} 道\n")

    rows = []
    for score, margin, bm in itertools.product(
            [0.50, 0.55, 0.60, 0.65], [0.015, 0.030, 0.050], [2.5, 5.0, 8.0]):
        object.__setattr__(cfg, "min_score", score)
        object.__setattr__(cfg, "min_margin", margin)
        object.__setattr__(cfg, "bm25_min", bm)
        recall, reject = evaluate(engine, items, cfg)
        rows.append((score, margin, bm, recall, reject))

    rows.sort(key=lambda r: -(r[3] * 0.45 + r[4] * 0.55))
    print(f"{'min_score':>9} {'margin':>7} {'bm25':>5} {'recall':>8} {'reject':>8} {'加权':>7}")
    for s, m, b, rec, rej in rows[:12]:
        print(f"{s:>9.2f} {m:>7.3f} {b:>5.1f} {rec:>7.1f}% {rej:>7.1f}% {rec*0.45+rej*0.55:>6.1f}")


if __name__ == "__main__":
    main()
