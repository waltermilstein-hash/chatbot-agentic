# =============================================================================
"""Single University AI app: serves the Chatbot (RAG) and Pegasus · autonomous
navigation (ReAct + LangGraph) in a single FastAPI process.

  · Chatbot      →  /virtualbot/chatbot/rag/chatbotservice/stream (SSE query)
                    /virtualbot/chatbot/rag/AutoTrainingBotByUser (training)
  · Pegasus · autonomous navigation →  /api/v1/agent/web  and  /api/v1/agent/web/stream (web search)

Runs OUTSIDE Docker; connects to the infrastructure (Weaviate/Mongo/Redis) at
127.0.0.7. No authentication: open endpoints (course demo on loopback)."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ChatbotService import api_router_chat_bot, init_openai_client
from agente_web.api import router as agente_router

app = FastAPI(
    title="University AI · Chatbot + 🐎 Pegasus · autonomous navigation",
    version="1.0",
    description="A single app: documental RAG (chatbot) + 🐎 Pegasus · autonomous navigation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
                   if os.getenv("CORS_ALLOWED_ORIGINS") else ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chatbot (RAG + training)
app.include_router(api_router_chat_bot, prefix="/virtualbot")
# Pegasus · autonomous navigation
app.include_router(agente_router, prefix="/api/v1")


@app.on_event("startup")
def _startup() -> None:
    log = logging.getLogger("unsl")
    try:
        init_openai_client()
    except Exception as exc:  # do not block startup if OPENAI_API_KEY is missing
        log.warning("init_openai_client failed: %s", exc)


@app.get("/health", tags=["infra"])
def health() -> dict:
    return {"status": "ok", "app": "unsl-ia", "components": ["chatbot", "agente-web"]}
