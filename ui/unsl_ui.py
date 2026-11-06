# =============================================================================
"""Single chainlit UI for University AI (single backend on :8085).

- Normal message  → RAG chatbot: /virtualbot/chatbot/rag/chatbotservice/stream (SSE)
- `/web <text>`   → Pegasus · autonomous navigation: /api/v1/agent/web/stream (SSE)

Shows an animated "🔎 Searching…" indicator (cl.Step, spinner) with live
progress while the stream arrives, then streams the answer token by token.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid

import chainlit as cl
import httpx

API_BASE   = os.getenv("UNSL_API_BASE", "http://127.0.0.1:8085")
CHAT_URL   = f"{API_BASE}/virtualbot/chatbot/rag/chatbotservice/stream"
WEB_URL    = f"{API_BASE}/api/v1/agent/web/stream"
TIMEOUT_S  = 180.0

USER    = os.getenv("UNSL_USER", "alumno@tesis.local")
COMPANY = os.getenv("UNSL_EMPRESA", "UNSL")

# INTENT router: decides web (Pegasus · autonomous navigation) vs documents (RAG)
# without the user having to type /web. Signals that the query is about the
# internet/current events → agent; otherwise RAG over the documents (default).
import re as _re
_WEB_INTENT = _re.compile(
    r"\b(noticia?s?|diario?s?|internet|en\s+la\s+web|buscá?\s+en\s+la\s+web|"
    r"[uú]ltim[oa]s?\s+(datos|noticias|novedades)|actualidad|hoy|ayer|"
    r"accidente|clima|tiempo|cotizaci[oó]n|precio\s+de|qu[eé]\s+pas[oó]|"
    r"resultado?s?\s+de|partido|elecciones|d[oó]lar|bolsa)\b",
    _re.IGNORECASE)


def _intent_es_web(text: str) -> bool:
    return bool(_WEB_INTENT.search(text or ""))


@cl.on_chat_start
async def start():
    cl.user_session.set("conversation_id", None)
    cl.user_session.set("thread_id", uuid.uuid4().hex)
    await cl.Message(content=(
        "✅ **University AI**\n"
        "- Type your query → **RAG chatbot** (indexed documents).\n"
        "- `/web <query>` → **🐎 Pegasus · autonomous navigation** (searches the internet)."
    )).send()


@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()
    forced_web = text.lower().startswith("/web")
    # Intent: explicit /web, or the classifier detects an internet query.
    is_web = forced_web or _intent_es_web(text)
    web_question = (text[4:] if forced_web else text).strip()

    if is_web:
        url = WEB_URL
        kw = {"json": {"question": web_question or "¿?",
                       "thread_id": cl.user_session.get("thread_id")}}
        step_label = "🐎 Pegasus · autonomous navigation — searching the web…"
    else:
        conv = cl.user_session.get("conversation_id")
        form = {"json_data": json.dumps({
            "user": USER, "company": COMPANY,
            "prompt": text, "timestamp": int(time.time()),
        }, ensure_ascii=False)}
        if conv:
            form["conversation_id"] = conv
            form["continuar_conversacion"] = "true"
        url = CHAT_URL
        kw = {"data": form}
        step_label = "📚 Searching documents…"

    answer = cl.Message(content="")
    sent = False
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            # cl.Step = animated indicator (spinner) with live progress
            async with cl.Step(name=step_label, type="tool") as step:
                step.output = "Connecting…"
                async with client.stream("POST", url, **kw) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode("utf-8", "replace")[:400]
                        step.output = f"HTTP {r.status_code}"
                        answer.content = f"❌ HTTP {r.status_code}: {body}"
                        await answer.send()
                        return
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        t = ev.get("type")
                        if t == "status":
                            step.output = ev.get("message", "…")          # ← "searching" animation
                        elif t in ("chunk", "content"):
                            txt = ev.get("text") or ev.get("message") or ""
                            if txt:
                                if not sent:
                                    await answer.send()
                                    sent = True
                                await answer.stream_token(txt)
                        elif t == "attachment":
                            # Generated file (docx/pdf/xlsx/pptx): download it with
                            # the token and attach it as a download in the chat.
                            durl = f"{API_BASE}{ev.get('download_url','')}"
                            fname = ev.get("filename", "export")
                            try:
                                rr = await client.get(durl, headers=kw.get("headers", {}))
                                if rr.status_code == 200:
                                    p = os.path.join(tempfile.gettempdir(), fname)
                                    with open(p, "wb") as fh:
                                        fh.write(rr.content)
                                    answer.elements = (answer.elements or []) + [
                                        cl.File(name=fname, path=p, display="inline")
                                    ]
                                    if not sent:
                                        await answer.send(); sent = True
                                    await answer.update()
                            except Exception as _e:  # noqa: BLE001
                                step.output = f"⚠️ Could not download the file: {_e}"
                        elif t == "metadata":
                            cid = ev.get("conversation_id")
                            if cid:
                                cl.user_session.set("conversation_id", cid)
                        elif t == "error":
                            msg = ev.get("user_message") or ev.get("message") or "error"
                            if not sent:
                                await answer.send(); sent = True
                            await answer.stream_token(f"\n\n⚠️ {msg}")
                step.output = "✅ Done"
        if not sent:
            answer.content = "ℹ️ No response."
            await answer.send()
        else:
            await answer.update()
    except httpx.ConnectError:
        answer.content = f"❌ Cannot connect to `{API_BASE}`. Is the backend up on :8085?"
        await answer.send()
    except Exception as exc:  # noqa: BLE001
        answer.content = f"❌ Error: `{exc}`"
        await answer.send()
