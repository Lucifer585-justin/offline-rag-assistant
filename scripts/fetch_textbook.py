"""从 GitHub 下载《计算机组成原理》分章讲义并合并成单本 PDF。"""

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(r"D:\ollama离线部署\offline-rag-assistant\data\docs\computer-organization-zh.pdf")
TMP = Path(r"D:\ollama离线部署\testdata\co_parts")
REPO = "YNUSharedHub/YNUSharePlatform"
SUBDIR = "软件学院/计算机组成原理"


def api(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ps"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2)


def safe_url(url: str) -> str:
    """GitHub 的 download_url 里中文和空格是原样的，直接传给 urlopen 会抛 InvalidURL。"""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe="/")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def download(url, dest, retries=3):
    """
    GitHub raw 直连经常超时，改用 curl（自带重试 + 断点），
    直连失败再挂 gh-proxy 镜像重试一次。
    """
    url = safe_url(url)
    candidates = [url, "https://gh-proxy.com/" + url]
    for attempt, target in enumerate(candidates):
        for i in range(retries):
            rc = subprocess.run(
                ["curl.exe", "--ssl-no-revoke", "-sS", "-L",
                 "--retry", "2", "--retry-delay", "2", "--max-time", "180",
                 "-A", "Mozilla/5.0", "-o", str(dest), target],
                capture_output=True).returncode
            if rc == 0 and dest.exists() and dest.stat().st_size > 10000:
                return dest.stat().st_size
            time.sleep(2)
        print(f"      直连失败，切换镜像…（{Path(url).name}）")
    raise RuntimeError(f"下载失败：{Path(url).name}")


TMP.mkdir(parents=True, exist_ok=True)
listing = api(f"https://api.github.com/repos/{REPO}/contents/{urllib.parse.quote(SUBDIR)}")
pdfs = sorted([f for f in listing if f["name"].endswith(".pdf")], key=lambda f: f["name"])
print(f"发现 {len(pdfs)} 个章节")

parts = []
for i, f in enumerate(pdfs, 1):
    dest = TMP / f"{i:02d}.pdf"
    size = download(f["download_url"], dest)
    parts.append(dest)
    print(f"  [{i}/{len(pdfs)}] {f['name']}  {size/1024:.0f} KB")

import pymupdf

merged = pymupdf.open()
for p in parts:
    with pymupdf.open(p) as src:
        merged.insert_pdf(src)
merged.set_metadata({
    "title": "计算机组成原理（课程讲义合集）",
    "author": "YNUSharedHub 课程资料",
    "subject": "Computer Organization course notes, compiled for offline RAG demo",
})
merged.save(str(OUT))
total_pages = merged.page_count
merged.close()

chars = 0
with pymupdf.open(OUT) as d:
    for page in d:
        chars += len(page.get_text("text"))

print(f"\n合并完成：{OUT}")
print(f"页数 {total_pages}，可抽取文本 {chars:,} 字符，文件 {OUT.stat().st_size/1024/1024:.1f} MB")
