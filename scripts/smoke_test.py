"""冒烟测试：健康检查 → 上传 PDF → 检索 → 多轮 → 删除。用法：python scripts/smoke_test.py"""

import http.cookiejar
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000"
ROOT = Path(__file__).resolve().parent.parent
CJ = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CJ))
OK, FAIL = 0, 0


def check(name, fn):
    global OK, FAIL
    try:
        detail = fn()
        OK += 1
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        FAIL += 1
        print(f"  [FAIL] {name}  ->  {type(exc).__name__}: {exc}")


def get_json(path):
    with OPENER.open(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post_json(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with OPENER.open(req, timeout=timeout) as r:
        return r.read().decode(), dict(r.headers)


UPLOADED = {"source": None}


def make_pdf(path: Path):
    """用 PyMuPDF 造一份含中文的测试 PDF，走一遍真实的 PDF 抽取路径。"""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open()
    page = doc.new_page()
    font = "C:/Windows/Fonts/msyh.ttc"
    text = (
        "第一章 总线与I/O\n\n"
        "总线按功能分为数据总线、地址总线和控制总线。\n"
        "总线带宽 = 总线宽度（字节） × 总线频率。例如 32 位宽、100 MHz 的总线，"
        "带宽 = 4 B × 100 M = 400 MB/s。\n\n"
        "I/O 控制方式有三种：程序查询方式、中断方式、DMA 方式。\n"
        "DMA 方式由 DMA 控制器直接在主存与外设之间传送数据块，仅在传送开始和结束时打扰 CPU。\n"
    )
    page.insert_text((50, 60), text, fontsize=11, fontname="msyh", fontfile=font)
    doc.save(str(path))
    doc.close()


print(f"== 冒烟测试 {BASE} ==")

check("GET /health", lambda: get_json("/api/health").get("ollama"))
check("GET /api/stats", lambda: f"chunks={get_json('/api/stats')['chunks']}")
check("GET / 首页", lambda: f"{len(OPENER.open(BASE + '/', timeout=15).read())} bytes")


def t_upload():
    pdf = ROOT / "data" / "docs" / "smoke-test.pdf"
    make_pdf(pdf)
    boundary = "----smoke" + str(int(time.time()))
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="smoke-test.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + "/api/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with OPENER.open(req, timeout=300) as r:
        data = json.loads(r.read().decode())
    pdf.unlink(missing_ok=True)
    assert data["ok"], data
    UPLOADED["source"] = data["source"]          # 记住真实落盘名（可能被重命名）
    return f"{data['source']} {data['pages']}p -> {data['chunks']} chunks"


check("POST /api/upload (PDF)", t_upload)


def t_hit():
    text, headers = post_json("/api/chat", {"message": "总线带宽怎么计算？32 位宽 100MHz 是多少？"})
    assert "400" in text or "MB/s" in text, text[:80]
    return f"{headers.get('X-Rag')} | {text[:40].strip()}"


def t_miss():
    text, headers = post_json("/api/chat", {"message": "量子纠缠是什么？"})
    assert text.startswith("知识库中未找到相关内容"), text[:60]
    return headers.get("X-Rag")


check("RAG 命中（总线带宽）", t_hit)
check("RAG 拒答（无关问题）", t_miss)


def t_memory():
    post_json("/api/chat", {"message": "我叫李四，正在准备计组考试"})
    text, _ = post_json("/api/chat", {"message": "我叫什么名字？"})
    assert "李四" in text, text[:60]
    return text.strip()[:30]


def t_reset():
    post_json("/api/reset", {})
    h = get_json("/api/history")
    assert h["turns"] == 0, h
    return "history cleared"


check("多轮记忆", t_memory)
check("清空记忆", t_reset)


def t_delete():
    docs = get_json("/api/documents")
    target = UPLOADED["source"]
    assert target, "上传步骤没有记录文件名"
    assert any(d["source"] == target for d in docs), f"not found in {[d['source'] for d in docs]}"
    req = urllib.request.Request(BASE + "/api/documents/" + target, method="DELETE")
    with OPENER.open(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    assert data["ok"], data
    return f"removed {data['removed']} chunks"


check("DELETE /api/documents", t_delete)

print(f"\n== 结果：通过 {OK} / {OK + FAIL} ==")
sys.exit(0 if FAIL == 0 else 1)
