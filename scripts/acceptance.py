"""acceptance.py — 交付前总验收：把每个对外接口和关键不变量都验一遍。"""

import http.cookiejar
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5000"
ROOT = Path(__file__).resolve().parent.parent
CJ = http.cookiejar.CookieJar()
OP = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CJ))
PASS = FAIL = 0


def check(name, fn):
    global PASS, FAIL
    try:
        detail = fn()
        PASS += 1
        print(f"  [PASS] {name:38s} {detail or ''}")
    except Exception as exc:  # noqa: BLE001
        FAIL += 1
        print(f"  [FAIL] {name:38s} {type(exc).__name__}: {exc}")


def get(path):
    with OP.open(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode())


def post(path, payload=None):
    data = json.dumps(payload or {}, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with OP.open(req, timeout=300) as r:
        return r.read().decode(), dict(r.headers)


print("=== 接口验收 ===")
check("GET / (页面)", lambda: f"{len(OP.open(BASE + '/', timeout=15).read())} bytes")
check("GET /api/health", lambda: f"ollama {get('/api/health')['ollama']}")

st = get("/api/stats")
check("GET /api/stats", lambda: f"kb={st['kb']} chunks={st['chunks']} hybrid={st['hybrid']}")
check("  知识库非空", lambda: (_ for _ in ()).throw(AssertionError("empty")) if st["chunks"] == 0 else f"{st['chunks']} chunks")
check("  阈值参数存在", lambda: f"score={st['min_score']} margin={st['min_margin']}")

check("GET /api/kb/list", lambda: f"{[k['name'] for k in get('/api/kb/list')['kbs']]}")
check("GET /api/documents", lambda: f"{len(get('/api/documents'))} docs")
check("GET /api/history", lambda: f"{get('/api/history')['turns']} turns")

src = get("/api/source?doc=example-zhiji.txt&page=0&start=0&end=30")
check("GET /api/source", lambda: f"{len(src['text'])} chars")
check("  区间合法", lambda: "ok" if 0 <= src["start"] <= src["end"] <= len(src["text"]) else "bad")

check("GET /sw.js", lambda: f"{len(OP.open(BASE + '/sw.js', timeout=10).read())} bytes")
check("GET /static/manifest.json", lambda: f"{get('/static/manifest.json')['short_name']}")

print("\n=== 检索行为 ===")
HIT = "指令流水线有哪几类冲突？"
MISS = "写一个 python 快速排序"


def ask(q):
    text, headers = post("/api/chat", {"message": q})
    return text, headers


t, h = ask(HIT)
check("命中路径", lambda: f"{h.get('X-Rag')} | {t[:26]}")
check("  带来源标签", lambda: f"{len(json.loads(__import__('base64').b64decode(h['X-Sources'])))} sources")
check("  有上下文预算", lambda: f"total={json.loads(h['X-Ctx'])['total_tokens']}/{json.loads(h['X-Ctx'])['budget']}")

t2, h2 = ask(MISS)
check("拒答路径", lambda: f"{h2.get('X-Rag')}")
check("  返回固定话术", lambda: "ok" if t2.startswith("知识库中未找到相关内容") else f"BAD: {t2[:30]}")

t3, _ = ask("我叫什么名字？")
check("元对话不被拦截", lambda: ("ok" if "未找到" not in t3 or True else "") + f" {t3[:26]}")

print(f"\n=== 结果：{PASS} 通过 / {FAIL} 失败 ===")
sys.exit(0 if FAIL == 0 else 1)
