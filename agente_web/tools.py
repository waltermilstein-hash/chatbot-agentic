# =============================================================================
# Project : University AI — Pegasus · autonomous navigation
# Module  : tools
# =============================================================================
"""Agent tools. Web search is performed with web search (OpenAI's native web
search, via the Responses API): the ReAct agent calls the `search_web` tool,
which in turn delegates navigation to OpenAI's search engine and returns the
result WITH the cited URLs (grounding)."""
from __future__ import annotations

import os

from langchain_core.tools import tool

SYSTEM = """You are an autonomous WEB RESEARCH AGENT.

How you work (ReAct):
- For ANY factual, current or verifiable data, ALWAYS use the `search_web` tool
  (which searches the web with web search). Never make up data, dates, figures
  or URLs: if the tool did not return it, do not assert it.
- You may chain several `search_web` calls (refining the query) until you have
  enough evidence.
- Answer in the user's language, concisely, and CITE every statement with its
  source (URL). If there is no evidence, say so instead of improvising.
"""


@tool
def search_web(query: str) -> str:
    """Search the web using web search (OpenAI's native web search).

    Returns a summary of what was found on the web plus the URLs of the cited
    sources. Use it for any data that must be verified or that is recent.
    `query` is the natural-language query.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tool_type = os.getenv("OPENAI_SEARCH_TOOL", "web_search_preview")
    model = os.getenv("SEARCH_MODEL", "gpt-4o-mini")

    try:
        resp = client.responses.create(
            model=model,
            tools=[{"type": tool_type}],
            input=query,
        )
    except Exception as exc:  # noqa: BLE001
        return f"[gpt-search error] {exc}"

    text = (getattr(resp, "output_text", "") or "").strip()

    # Collect the citations (url_citation annotations) to keep the sources.
    cites: list[str] = []
    for item in (getattr(resp, "output", None) or []):
        for part in (getattr(item, "content", None) or []):
            for ann in (getattr(part, "annotations", None) or []):
                if getattr(ann, "type", "") == "url_citation":
                    url = getattr(ann, "url", "") or ""
                    title = getattr(ann, "title", "") or url
                    if url:
                        cites.append(f"- {title} — {url}")

    if cites:
        text += "\n\nSources:\n" + "\n".join(dict.fromkeys(cites))
    return text or "No results for that query."


TOOLS = [search_web]
