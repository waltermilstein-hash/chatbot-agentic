# =============================================================================
# Project : University AI — Pegasus · autonomous navigation
# Module  : api
# =============================================================================
"""Pegasus · autonomous navigation API. Exposes autonomous search (synchronous
and SSE streaming)."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agente_web import agent as agent_mod

router = APIRouter(tags=["Pegasus · autonomous navigation"])


class InvestigarRequest(BaseModel):
    question: str = Field(..., description="Natural-language query to research on the web.")
    thread_id: Optional[str] = Field(None, description="Same id across successive turns → multi-turn memory.")


class InvestigarResponse(BaseModel):
    answer: str
    steps: list = Field(default_factory=list)


@router.post("/agent/web", response_model=InvestigarResponse, summary="Research on the web (synchronous)")
async def investigate(req: InvestigarRequest) -> InvestigarResponse:
    out = await agent_mod.investigate(req.question, req.thread_id)
    return InvestigarResponse(**out)


@router.post("/agent/web/stream", summary="Research on the web (SSE streaming)")
async def investigate_stream(req: InvestigarRequest):
    async def gen():
        yield f"data: {json.dumps({'type': 'status', 'message': '🔎 Researching...'})}\n\n"
        try:
            async for tok in agent_mod.investigate_stream(req.question, req.thread_id):
                yield f"data: {json.dumps({'type': 'chunk', 'text': tok}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
