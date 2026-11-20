"""
app.py — Flask 入口（路由层，只负责编排，不写业务细节）

分层：
    app.py        路由 + 上下文组装 + 流式转发      ← 本文件
    rag_engine.py 抽取/切块/编码/检索/文档管理
    memory.py     会话与消息存取
    config.py     配置
    logger.py     日志

一次 /api/chat 的 7 步：
    ① Cookie 取 sid
    ② 取历史（memory）
    ③ 组检索 query（必要时拼接上一轮，处理指代）
    ④ 检索 Top-K（rag_engine）
    ⑤ 预算裁剪后拼 messages（本文件 build_messages）
    ⑥ 流式调 Ollama
    ⑦ 落库（finally，客户端断开也留痕）
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from logger import get_logger, setup_logging
from memory import SessionStore
from rag_engine import RAGEngine, RAGError, SUPPORTED_EXT

log = get_logger(__name__)

# ------------------------------------------------------------------------------
# 提示词：三套，对应三种场景
# ------------------------------------------------------------------------------
SYSTEM_RAG = """你是《计算机组成原理》课程的助教。请严格依据【资料片段】和【历史对话】回答学生的问题。

必须遵守的规则：
1. 只能使用【资料片段】中出现的信息回答课程知识问题，不得使用片段之外的任何知识，不得推测、不得补充。
2. 如果【资料片段】中没有能回答该问题的内容，只输出这一句话：知识库中未找到相关内容。
3. 引用片段时在句末标注编号，如 [1]、[2]；只能标注真实存在的编号。
4. 用中文作答：先给结论，再分点说明；公式与数值必须与片段完全一致。
5. 若学生问的是对话本身（"我刚才问了什么"），可以依据【历史对话】回答。

【资料片段】
{context}
"""

SYSTEM_NO_DOC = """你是运行在本机的课程助教。本次没有检索到任何资料片段。

规则（严格执行）：
1. 你**只能**依据【历史对话】回答关于对话本身的问题：学生的名字、刚才聊过什么、上一句说了什么。
2. 除上述之外，任何知识性问题（无论是否属于本课程：物理、编程、常识、公式、定义）都必须只回复这一句：
   知识库中未找到相关内容。
3. 禁止使用你的参数化知识作答，禁止解释、禁止补充、禁止给例子。
4. 用中文，简洁。
"""

SYSTEM_PLAIN = "你是运行在本地电脑上的端侧 AI 助手，网络已断开。回答简洁、直接、可执行。"

NO_HIT_ANSWER = "知识库中未找到相关内容。"
ANAPHORA = ("那", "它", "这个", "那个", "上述", "刚才", "上面", "该", "其", "这种", "此")


# ==============================================================================
# 上下文预算
# ==============================================================================
def est_tokens(text: str) -> int:
    """不依赖 tokenizer 的粗估：中文 1 字≈1 token，其他 3 字符≈1 token，误差 ±15%。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + max(1, (len(text) - cjk) // 3)


def fit_history(history: list[dict], budget: int) -> tuple[list[dict], int, int]:
    """
    从最新往最旧累加，装不下就停（保住最近的对话）。
    返回 (保留历史, 消耗 token, 丢弃条数)。
    丢完要保证第一条是 user —— 否则模型会看到一个没有问题的回答。
    """
    kept, used = [], 0
    for msg in reversed(history):
        cost = est_tokens(msg["content"]) + 8
        if used + cost > budget and kept:
            break
        kept.append(msg)
        used += cost
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        used -= est_tokens(kept[0]["content"]) + 8
        kept.pop(0)
    return kept, used, len(history) - len(kept)


def build_messages(cfg: Config, history: list[dict], hits: list[dict], question: str,
                   kb_ready: bool = True):
    """
    组装最终 messages，并返回预算明细（前端 F12 可看）。
    优先级：当前问题 > RAG 片段 > 历史。

    三种模式：
      · 命中       → SYSTEM_RAG（只能照片段答）
      · 未命中但库可用 → SYSTEM_NO_DOC（只准回答对话本身）
      · 库为空     → SYSTEM_PLAIN（纯聊天助手，方便首次使用先试试模型）
    """
    budget = cfg.prompt_budget
    q_tokens = est_tokens(question)
    fixed = 260 + q_tokens          # system 规则文本 + 本轮问题

    if hits:
        blocks, ctx_tokens, used_hits = [], 0, []
        for i, h in enumerate(hits[: cfg.top_k], 1):
            block = f"[{i}] 来源：{h['source']}" + (f" 第 {h['page']} 页" if h.get("page") else "")
            block += f"\n{h['text']}"
            cost = est_tokens(block)
            # RAG 最多占到预算的 70%，剩下留给历史
            if blocks and fixed + ctx_tokens + cost > budget * 0.7:
                break
            blocks.append(block)
            ctx_tokens += cost
            used_hits.append(h)
        system_prompt, hits = SYSTEM_RAG.format(context="\n\n".join(blocks)), used_hits
    else:
        system_prompt = SYSTEM_NO_DOC if kb_ready else SYSTEM_PLAIN
        ctx_tokens = 0

    sys_tokens = est_tokens(system_prompt)
    hist_budget = max(0, budget - sys_tokens - fixed + 260)
    kept, hist_tokens, dropped = fit_history(history, hist_budget)

    messages = [{"role": "system", "content": system_prompt}]
    messages += kept
    messages.append({"role": "user", "content": question})

    meta = {
        "question_tokens": q_tokens, "system_tokens": sys_tokens,
        "context_tokens": ctx_tokens, "history_tokens": hist_tokens,
        "history_kept": len(kept) // 2, "history_dropped": dropped,
        "total_tokens": sys_tokens + ctx_tokens + hist_tokens + q_tokens,
        "budget": budget,
    }
    return messages, hits, meta


# ==============================================================================
# 应用工厂
# ==============================================================================
def create_app(cfg: Config | None = None) -> Flask:
    cfg = cfg or Config.load()
    setup_logging(cfg.log_dir, cfg.log_level)
    log.info("启动配置：%s", json.dumps(cfg.summary(), ensure_ascii=False))

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_bytes
    app.config["JSON_AS_ASCII"] = False

    engine = RAGEngine(cfg)
    store = SessionStore(cfg.sessions_path)
    store.prune(keep_days=30)

    # ------------------------------------------------------------------ 辅助
    def current_sid(create: bool = True) -> tuple[str | None, bool]:
        sid = request.cookies.get("sid")
        if sid and store.exists(sid):
            return sid, False
        if not create:
            return None, False
        sid = store.new_sid()
        store.touch(sid)
        return sid, True

    def with_sid(resp: Response, sid: str | None, is_new: bool) -> Response:
        if sid:
            # HttpOnly 降低 XSS 盗用风险；SameSite=Lax 阻止跨站请求携带
            resp.set_cookie("sid", sid, max_age=30 * 86400, httponly=True, samesite="Lax")
        return resp

    # ------------------------------------------------------------------ PWA
    @app.get("/sw.js")
    def service_worker():
        """
        Service Worker 必须从根路径提供，作用域才是 /；
        放在 /static/sw.js 的话 scope 会被限制在 /static/，页面收不到控制。
        """
        from flask import send_from_directory
        resp = send_from_directory(Path(app.root_path) / "static", "sw.js",
                                   mimetype="application/javascript")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    # ------------------------------------------------------------------ 页面
    @app.get("/")
    def index():
        sid, is_new = current_sid()
        resp = Response(render_template(
            "index.html",
            model=cfg.model,
            embed_model=cfg.embed_model,
            num_ctx=cfg.num_ctx,
            max_upload_mb=cfg.max_upload_mb,
        ), mimetype="text/html")
        return with_sid(resp, sid, is_new)

    # ------------------------------------------------------------------ 状态
    @app.get("/api/health")
    def api_health():
        try:
            with urllib.request.urlopen(f"{cfg.ollama_base}/api/version", timeout=3) as r:
                version = json.loads(r.read().decode())["version"]
            return jsonify(ok=True, ollama=version, model=cfg.model)
        except Exception as exc:  # noqa: BLE001
            log.warning("健康检查失败：%s", exc)
            return jsonify(ok=False, error=str(exc)), 503

    @app.get("/api/stats")
    def api_stats():
        sid, _ = current_sid(create=False)
        turns = store.count(sid) // 2 if sid else 0
        st = engine.stats()
        st.update({
            "turns": turns,
            "model": cfg.model,
            "num_ctx": cfg.num_ctx,
            "prompt_budget": cfg.prompt_budget,
        })
        return jsonify(st)

    @app.get("/api/history")
    def api_history():
        sid, _ = current_sid(create=False)
        if not sid:
            return jsonify(messages=[], turns=0)
        msgs = store.history(sid, limit=40)
        return jsonify(messages=msgs, turns=len(msgs) // 2)

    @app.post("/api/reset")
    def api_reset():
        sid, is_new = current_sid()
        store.clear(sid) if sid else None
        return with_sid(jsonify(ok=True), sid, is_new)

    # ------------------------------------------------------------------ 文档
    @app.post("/api/upload")
    def api_upload():
        """上传文档 → 自动解析、切块、入库。不需要用户改 docs 目录再跑脚本。"""
        if "file" not in request.files:
            return jsonify(ok=False, error="没有收到文件（字段名应为 file）"), 400

        f = request.files["file"]
        raw_name = f.filename or ""
        ext = Path(raw_name).suffix.lower()
        if ext not in SUPPORTED_EXT:
            return jsonify(ok=False, error=f"只支持 {', '.join(sorted(SUPPORTED_EXT))}"), 400

        # 只保留安全字符：中文文件名必须保留，但要挡住 ../ 和 Windows 非法字符
        stem = Path(raw_name).stem
        safe_stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem).strip(". ") or "document"
        filename = f"{safe_stem}{ext}"
        target = Path(cfg.docs_dir) / filename
        counter = 1
        while target.exists():
            filename = f"{safe_stem}_{counter}{ext}"
            target = Path(cfg.docs_dir) / filename
            counter += 1

        try:
            f.save(target)
            result = engine.add_document(target)
            log.info("上传成功：%s → %d 块", filename, result["chunks"])
            return jsonify(ok=True, **result, stats=engine.stats())
        except RAGError as exc:
            target.unlink(missing_ok=True)
            log.warning("上传失败 %s：%s", filename, exc)
            return jsonify(ok=False, error=str(exc)), 400
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            log.exception("上传异常 %s", filename)
            return jsonify(ok=False, error=f"处理失败：{exc}"), 500

    @app.get("/api/source")
    def api_source():
        """
        取某文档某页的原文，供前端做引用高亮。
        只返回清洗后的页文本 + 字符区间，前端自己切片，避免把 HTML 从这里吐出去。
        """
        doc = request.args.get("doc", "")
        try:
            page = int(request.args.get("page", 0))
            start = int(request.args.get("start", 0))
            end = int(request.args.get("end", 0))
        except ValueError:
            return jsonify(ok=False, error="参数必须是整数"), 400

        text = engine.page_text(doc, page)
        if text is None:
            return jsonify(ok=False, error=f"找不到 {doc} 第 {page} 页"), 404

        # 区间兜底：老数据可能没有 offset，或超出范围
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        return jsonify(ok=True, source=doc, page=page, start=start, end=end, text=text)

    @app.get("/api/documents")
    def api_documents():
        return jsonify(engine.list_documents())

    @app.delete("/api/documents/<path:source>")
    def api_delete_document(source: str):
        try:
            removed = engine.delete_document(source)
            return jsonify(ok=True, removed=removed, stats=engine.stats())
        except Exception as exc:  # noqa: BLE001
            log.exception("删除文档失败：%s", source)
            return jsonify(ok=False, error=str(exc)), 500

    @app.post("/api/rebuild")
    def api_rebuild():
        """把 docs 目录整个重新入库（换 embedding 模型后必须做）。"""
        try:
            result = engine.build_from_dir(reset=True)
            return jsonify(ok=result["ok"], **result, stats=engine.stats())
        except Exception as exc:  # noqa: BLE001
            log.exception("重建知识库失败")
            return jsonify(ok=False, error=str(exc)), 500

    # ------------------------------------------------------------------ 对话
    @app.post("/api/chat")
    def api_chat():
        data = request.get_json(silent=True) or {}
        question = (data.get("message") or "").strip()
        if not question:
            return jsonify(ok=False, error="消息不能为空"), 400
        if len(question) > 4000:
            # 4000 字 ≈ 4000 token，已吃掉 prompt 预算（6992）的一半以上，
            # 再长就会把 RAG 片段和历史全部挤掉，甚至超 num_ctx 被模型静默截断
            return jsonify(ok=False, error="问题过长（上限 4000 字，约 4000 token）"), 400

        sid, is_new = current_sid()
        history = store.history(sid, limit=40) if sid else []

        # 指代消解：只在"问题自身不足以表达意图"时才拼接上一轮
        #   a) 含明确指代词（那 / 它 / 这个 / 上述 / 刚才 …）
        #   b) 极短（≤6 字）且不含疑问实词 —— "展开说说""继续"这种
        # 反例：8 字的"量子纠缠是什么？"不该拼接，否则会被上一轮的"计组考试"拉进知识库
        prev_user = ""
        for msg in reversed(history):
            if msg["role"] == "user":
                prev_user = msg["content"]
                break
        QUESTION_WORDS = ("什么", "怎么", "多少", "哪", "谁", "为什么", "如何", "是否", "区别")
        explicit_pronoun = any(k in question[:15] for k in ANAPHORA)
        short_followup = len(question) <= 6 and not any(w in question for w in QUESTION_WORDS)
        need_context = bool(prev_user) and (explicit_pronoun or short_followup)
        query = f"{prev_user[-120:]} {question}" if need_context else question

        hits, best = [], 0.0
        try:
            hits, best = engine.search(query)
        except RAGError as exc:
            log.warning("检索降级：%s", exc)
        except Exception:  # noqa: BLE001
            log.exception("检索异常，降级为纯聊天")

        # ------------------------------------------------------------------
        # 防幻觉硬闸门：没检索到 + 不是"元对话"问题 → 模型根本不参与
        #
        # 为什么不能只靠提示词：实测 7B 会把"量子纠缠是什么"归类成"非本课程知识"，
        # 于是绕过"禁止回答课程知识"的规则，直接凭参数知识作答。
        # 唯一可靠的办法是架构上不调用它。
        #
        # 元对话（"我叫什么""刚才问了什么"）必须放行，否则多轮记忆就没意义了 ——
        # 这类问题同样检索不到，但答案在历史里。
        # ------------------------------------------------------------------
        META_PATTERNS = ("我叫", "我的名字", "我是谁", "我刚才", "我之前", "我说过",
                         "上一个问题", "刚才问", "刚才说", "记得", "上面说", "你刚才", "我问了")
        is_meta = any(p in question for p in META_PATTERNS)

        if not hits and not is_meta and engine.ready:
            log.info("拒答（无召回）：%s | best=%.3f", question[:40], best)
            if sid:
                try:
                    store.add_turn(sid, question, NO_HIT_ANSWER)
                    store.touch(sid, turns_delta=1)
                except Exception:  # noqa: BLE001
                    log.exception("写入拒答记录失败")
            headers = {
                "X-Sources": base64.b64encode(b"[]").decode(),
                "X-Rag": f"miss;best={best:.3f}",
                "X-Ctx": json.dumps({"blocked_by": "retrieval_gate"}, ensure_ascii=False),
            }
            return with_sid(Response(NO_HIT_ANSWER, mimetype="text/plain; charset=utf-8",
                                     headers=headers), sid, is_new)

        messages, hits, meta = build_messages(cfg, history, hits, question, kb_ready=engine.ready)
        payload = {
            "model": cfg.model,
            "stream": True,
            "messages": messages,
            "options": {"temperature": 0.1, "num_ctx": cfg.num_ctx},
        }

        sources = [
            {"n": i, "source": h["source"], "page": h["page"], "score": h["score"],
             "start": h.get("char_start", 0), "end": h.get("char_end", 0),
             "preview": h["text"][:120].replace("\n", " ")}
            for i, h in enumerate(hits, 1)
        ]

        def generate():
            parts: list[str] = []
            req = urllib.request.Request(
                f"{cfg.ollama_base}/api/chat",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = obj.get("message", {}).get("content", "")
                        if piece:
                            parts.append(piece)
                            yield piece
                        if obj.get("done"):
                            break
            except urllib.error.URLError as exc:
                msg = f"\n[后端错误] 连不上 Ollama（{cfg.ollama_base}）：{exc.reason}"
                parts.append(msg)
                yield msg
            except Exception as exc:  # noqa: BLE001
                log.exception("生成失败")
                msg = f"\n[后端错误] {type(exc).__name__}: {exc}"
                parts.append(msg)
                yield msg
            finally:
                if sid:
                    try:
                        store.add_turn(sid, question, "".join(parts).strip() or "(空)")
                        store.touch(sid, turns_delta=1)
                    except Exception:  # noqa: BLE001
                        log.exception("写入会话失败")

        headers = {
            "X-Sources": base64.b64encode(json.dumps(sources, ensure_ascii=False).encode()).decode(),
            "X-Rag": f"{'hit' if hits else 'miss'};best={best:.3f}",
            "X-Ctx": json.dumps(meta, ensure_ascii=False),
        }
        return with_sid(Response(generate(), mimetype="text/plain; charset=utf-8",
                                headers=headers), sid, is_new)

    # ------------------------------------------------------------------ 错误
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        return jsonify(ok=False, error=f"文件超过 {cfg.max_upload_mb} MB 上限"), 413

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify(ok=False, error="接口不存在"), 404

    @app.errorhandler(500)
    def server_error(e):  # noqa: BLE001
        log.exception("未捕获异常：%s", e)
        return jsonify(ok=False, error="服务内部错误，详见 logs/app.log"), 500

    app.extensions["cfg"] = cfg
    app.extensions["engine"] = engine
    app.extensions["store"] = store
    return app


# ==============================================================================
# 入口
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="离线 RAG 助教")
    parser.add_argument("--host", default=None, help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认 5000")
    parser.add_argument("--rebuild", action="store_true", help="启动前把 data/docs 重新入库")
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动预热")
    args = parser.parse_args()

    cfg = Config.load()
    setup_logging(cfg.log_dir, cfg.log_level)
    app = create_app(cfg)
    engine: RAGEngine = app.extensions["engine"]

    if args.rebuild:
        log.info("重建知识库…")
        result = engine.build_from_dir(reset=True)
        log.info("重建完成：%d 篇 / %d 块", len(result["documents"]), result["chunks"])

    stats = engine.stats()
    log.info("知识库：%d 块 / %d 文件 / %d 维%s",
             stats["chunks"], stats["documents"], stats["dim"],
             f"（{stats['error']}）" if stats["error"] else "")

    host, port = args.host or cfg.host, args.port or cfg.port
    log.info("监听 http://%s:%d  model=%s  RAG=%s", host, port, cfg.model,
             "on" if stats["ready"] else "off")
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
