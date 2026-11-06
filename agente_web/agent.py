# =============================================================================
# Project : University AI — Pegasus · autonomous navigation
# Module  : agent
# =============================================================================
"""ReAct agent on top of LangGraph: the LLM decides which tool to call
(`search_web` / `read_page`), reads what is returned and synthesizes with
citations.

Optional multi-turn memory per `thread_id` (in-memory checkpointer)."""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agente_web.tools import SYSTEM, TOOLS

_checkpointer = MemorySaver()


@lru_cache(maxsize=1)
def _model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_agent():
    """ReAct agent with memory shared by thread_id."""
    return create_react_agent(_model(), TOOLS, prompt=SYSTEM, checkpointer=_checkpointer)


def _cfg(thread_id: str | None) -> dict:
    return {"configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": int(os.getenv("AGENT_RECURSION_LIMIT", "25"))}


async def investigate(question: str, thread_id: str | None = None) -> dict:
    """Run the agent and return the final answer plus the steps (tool calls)."""
    agent = get_agent()
    result = await agent.ainvoke({"messages": [("user", question)]}, config=_cfg(thread_id))
    msgs = result["messages"]
    answer = msgs[-1].content if msgs else ""
    steps = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            steps.append({"tool": tc.get("name"), "args": tc.get("args", {})})
    return {"answer": answer, "steps": steps}


async def investigate_stream(question: str, thread_id: str | None = None):
    """Async generator: yields tokens of the agent's final message."""
    agent = get_agent()
    async for chunk, _meta in agent.astream(
        {"messages": [("user", question)]}, config=_cfg(thread_id), stream_mode="messages"
    ):
        # Only assistant text tokens (not the tool calls).
        text = getattr(chunk, "content", "")
        if text and getattr(chunk, "type", "") in ("ai", "AIMessageChunk"):
            yield text
