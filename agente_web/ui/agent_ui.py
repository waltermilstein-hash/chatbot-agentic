# =============================================================================
# Project : University AI — chainlit UI for Pegasus · autonomous navigation
# =============================================================================
"""Chat UI (chainlit) for Pegasus · autonomous navigation. Talks to the backend via:
  POST {AGENT_API_BASE}/api/v1/agent/web/stream   (SSE)

Run (from agente-web/app, with the agent's venv):
    chainlit run ui/agent_ui.py --port 8503
"""
from __future__ import annotations

import json
import os
import uuid

import chainlit as cl
import httpx

API_BASE  = os.getenv("AGENT_API_BASE", "http://127.0.0.1:8080")
STREAM_URL = f"{API_BASE}/api/v1/agent/web/stream"
TIMEOUT_S = 180.0


@cl.on_chat_start
async def start():
    cl.user_session.set("thread_id", uuid.uuid4().hex)
    await cl.Message(content="🔎 **Pegasus · autonomous navigation**. Ask something and I'll search the web citing sources.").send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    payload = {"question": message.content, "thread_id": thread_id}

    out = cl.Message(content="")
    await out.send()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            async with client.stream("POST", STREAM_URL, json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "replace")[:500]
                    out.content = f"❌ HTTP {r.status_code}: {body}"
                    await out.update()
                    return
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        ev = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("type")
                    if t == "chunk":
                        await out.stream_token(ev.get("text", ""))
                    elif t == "error":
                        await out.stream_token(f"\n\n⚠️ {ev.get('message', 'error')}")
        await out.update()
    except httpx.ConnectError:
        out.content = f"❌ Cannot connect to the agent at `{STREAM_URL}`. Did you start main_api?"
        await out.update()
    except Exception as exc:  # noqa: BLE001
        out.content = f"❌ Error: `{exc}`"
        await out.update()
