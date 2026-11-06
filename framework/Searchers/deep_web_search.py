# =============================================================================
# =============================================================================
"""
Deep Web Search provider for the Deep Research tool.

Exposes a single async entry point:

    async def search(query: str, max_results: int = 5) -> list[dict]
        -> [{"title": str, "url": str, "snippet": str}, ...]

returning REAL sources with REAL URLs (not a synthesized HTML blob).
Imported by the orchestrator as ``framework.Searchers.deep_web_search.search``.

────────────────────────────────────────────────────────────────────────────
PROVIDER SELECTION (first available wins; override with DEEP_WEB_SEARCH_PROVIDER)
────────────────────────────────────────────────────────────────────────────
1. "openai"  (ACTIVE in this environment) — uses ``Constants.KEY_OPEN_AI`` with
   the OpenAI Responses API and the built-in ``web_search`` tool. The model's
   answer carries ``url_citation`` annotations (real title + real url); we map
   each citation to {title, url, snippet} and build the snippet from the cited
   passage of the answer text. No extra API key required, so this is preferred.

   NOTE: the chat-completions model ``gpt-5.6-search-preview`` named in the spec
   returns HTTP 404 ("model does not exist or you do not have access") against
   the live OpenAI key in this environment, so it is NOT used. The equivalent,
   working OpenAI web-search mechanism is the Responses API ``web_search`` tool,
   which this module uses. The search model is configurable via
   ``DEEP_WEB_SEARCH_MODEL`` (default: Constants.GPT_5_1 = "gpt-5.1").

2. Dedicated search API via httpx (async) IF its env key is present, checked in
   order: TAVILY_API_KEY, SERPAPI_API_KEY, BING_SEARCH_KEY,
   (GOOGLE_CSE_KEY|GOOGLE_CSE_API_KEY)+GOOGLE_CSE_ID. Each is mapped to the
   {title, url, snippet} shape. None of these keys are set here.

3. Degraded fallback — wraps the legacy ``framework.Searchers.SearchInternet``
   WebSearchClient and returns a single-item list
   ``[{"title": ..., "url": "", "snippet": <text>}]`` so callers still get
   something. Used only when no OpenAI key and no dedicated key is available.

────────────────────────────────────────────────────────────────────────────
HOW TO SWITCH PROVIDERS (env vars)
────────────────────────────────────────────────────────────────────────────
- DEEP_WEB_SEARCH_PROVIDER = openai | tavily | serpapi | bing | google | degraded
      Force a specific provider (otherwise auto-selected in the order above).
- DEEP_WEB_SEARCH_MODEL     OpenAI model for the web_search tool (default gpt-5.1).
- TAVILY_API_KEY            Enable Tavily.
- SERPAPI_API_KEY           Enable SerpAPI (google engine).
- BING_SEARCH_KEY           Enable Bing Web Search v7.
- GOOGLE_CSE_KEY / GOOGLE_CSE_API_KEY  +  GOOGLE_CSE_ID   Enable Google CSE.

Defensive contract: ANY provider error is swallowed and ``search`` returns ``[]``
(it never raises). Results are deduped by url, snippets trimmed to ~300 chars,
and at most ``max_results`` items are returned.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List, Dict

try:
    from config.Constants import KEY_OPEN_AI, GPT_5_1
except Exception:  # pragma: no cover - config must exist in the real app
    KEY_OPEN_AI = os.getenv("KEY_OPEN_AI", "")
    GPT_5_1 = "gpt-5.1"

# ── Tunables ──────────────────────────────────────────────────────────────
_SNIPPET_MAX = 300
_OPENAI_TIMEOUT = float(os.getenv("DEEP_WEB_SEARCH_TIMEOUT", "150"))
_HTTP_TIMEOUT = float(os.getenv("DEEP_WEB_SEARCH_HTTP_TIMEOUT", "20"))
_DEFAULT_OPENAI_MODEL = os.getenv("DEEP_WEB_SEARCH_MODEL") or GPT_5_1


# ── Public API ──────────────────────────────────────────────────────────────
async def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Return real web-search results for ``query``.

    Args:
        query: the search query / research question.
        max_results: maximum number of sources to return.

    Returns:
        A list of ``{"title", "url", "snippet"}`` dicts. Empty list on any error
        or when nothing is found. Never raises.
    """
    try:
        if not query or not str(query).strip():
            return []
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        if max_results <= 0:
            return []

        provider = _select_provider()

        if provider == "openai":
            raw = await _search_openai(query, max_results)
        elif provider == "tavily":
            raw = await _search_tavily(query, max_results)
        elif provider == "serpapi":
            raw = await _search_serpapi(query, max_results)
        elif provider == "bing":
            raw = await _search_bing(query, max_results)
        elif provider == "google":
            raw = await _search_google_cse(query, max_results)
        else:  # "degraded"
            raw = await _search_degraded(query)

        return _normalize(raw, max_results)
    except Exception:
        # Fully defensive: callers must always get a list.
        return []


# ── Provider selection ───────────────────────────────────────────────────────
def _select_provider() -> str:
    forced = (os.getenv("DEEP_WEB_SEARCH_PROVIDER") or "").strip().lower()
    if forced:
        return forced

    if KEY_OPEN_AI:
        return "openai"
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    if os.getenv("SERPAPI_API_KEY"):
        return "serpapi"
    if os.getenv("BING_SEARCH_KEY"):
        return "bing"
    if (os.getenv("GOOGLE_CSE_KEY") or os.getenv("GOOGLE_CSE_API_KEY")) and os.getenv("GOOGLE_CSE_ID"):
        return "google"
    return "degraded"


# ── Provider 1: OpenAI Responses API + web_search tool ────────────────────────
async def _search_openai(query: str, max_results: int) -> List[Dict[str, str]]:
    """Use the OpenAI web_search tool and extract url_citation annotations.

    Preferred because it needs no extra API key (same KEY_OPEN_AI as the rest of
    the pipeline). Falls back to parsing a JSON list from the model output if no
    citation annotations are present.
    """
    from openai import AsyncOpenAI  # local import: optional dependency boundary

    client = AsyncOpenAI(api_key=KEY_OPEN_AI)

    prompt = (
        f"Search the web and find up to {max_results} relevant, high-quality "
        f"sources that answer this query:\n\n\"{query}\"\n\n"
        "Write a short factual answer and cite each source with its web page "
        "so that citations are attached to the relevant sentences."
    )

    resp = await asyncio.wait_for(
        client.responses.create(
            model=_DEFAULT_OPENAI_MODEL,
            tools=[{"type": "web_search"}],
            reasoning={"effort": "low"},
            input=prompt,
            max_output_tokens=4000,
        ),
        timeout=_OPENAI_TIMEOUT,
    )

    results = _extract_citations_from_responses(resp)
    if results:
        return results

    # No annotations: ask the model to emit structured JSON as a robust fallback.
    return _parse_json_results(getattr(resp, "output_text", "") or "")


def _extract_citations_from_responses(resp) -> List[Dict[str, str]]:
    """Walk a Responses API result and pull url_citation annotations.

    Each annotation has ``url`` + ``title`` + ``start_index``/``end_index`` into
    the enclosing text part; we slice that text to build a snippet.
    """
    out: List[Dict[str, str]] = []
    output_items = getattr(resp, "output", None) or []
    for item in output_items:
        content_parts = getattr(item, "content", None) or []
        for part in content_parts:
            text = getattr(part, "text", None) or ""
            annotations = getattr(part, "annotations", None) or []
            for ann in annotations:
                data = _as_dict(ann)
                if data.get("type") not in (None, "url_citation"):
                    continue
                url = (data.get("url") or "").strip()
                if not url:
                    continue
                title = (data.get("title") or "").strip() or url
                snippet = _slice_snippet(text, data.get("start_index"), data.get("end_index"))
                out.append({"title": title, "url": url, "snippet": snippet})
    return out


# Markdown inline-citation token emitted by the web_search tool, e.g.
# "([example.com](https://example.com/x))" or a bare "[label](url)".
_CITATION_TOKEN_RE = re.compile(r"\(?\[[^\]]*\]\((?:https?://)?[^)]*\)\)?")


def _clean_citation_markers(text: str) -> str:
    """Strip inline markdown citation tokens and tidy leftover whitespace."""
    cleaned = _CITATION_TOKEN_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _slice_snippet(text: str, start, end) -> str:
    """Build a readable snippet from the prose a citation supports.

    The url_citation span ``[start:end]`` is usually the inline citation marker
    itself, so we take the prose that *precedes* it (the sentence the source
    backs), clean out any markdown citation tokens, and trim to ~300 chars.
    """
    if not text:
        return ""
    try:
        start = int(start)
        end = int(end)
    except (TypeError, ValueError):
        return _clean_citation_markers(text)[:_SNIPPET_MAX]

    if start < 0 or start > len(text):
        start = 0
    if end <= 0 or end > len(text):
        end = len(text)

    preceding = _clean_citation_markers(text[:start])
    if len(preceding) >= 40:
        return preceding[-_SNIPPET_MAX:].strip()

    window = _clean_citation_markers(text[:end])
    if window:
        return window[-_SNIPPET_MAX:].strip()
    return _clean_citation_markers(text)[:_SNIPPET_MAX]


def _parse_json_results(text: str) -> List[Dict[str, str]]:
    """Best-effort parse of a JSON ``[{title,url,snippet}]`` list from model text."""
    if not text:
        return []
    candidate = text.strip()
    if "```" in candidate:
        candidate = re.sub(r"```(?:json)?", "", candidate).replace("```", "").strip()
    match = re.search(r"\[.*\]", candidate, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return []
    out: List[Dict[str, str]] = []
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            url = (row.get("url") or "").strip()
            if not url:
                continue
            title = (row.get("title") or "").strip() or url
            snippet = (row.get("snippet") or row.get("description") or "").strip()
            out.append({"title": title, "url": url, "snippet": snippet})
    return out


# ── Provider 2: dedicated HTTP search APIs (httpx, async) ─────────────────────
async def _search_tavily(query: str, max_results: int) -> List[Dict[str, str]]:
    import httpx

    key = os.getenv("TAVILY_API_KEY")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
        r = await http.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
        )
        r.raise_for_status()
        data = r.json()
    out: List[Dict[str, str]] = []
    for row in (data.get("results") or []):
        url = (row.get("url") or "").strip()
        if not url:
            continue
        out.append({
            "title": (row.get("title") or "").strip() or url,
            "url": url,
            "snippet": (row.get("content") or "").strip(),
        })
    return out


async def _search_serpapi(query: str, max_results: int) -> List[Dict[str, str]]:
    import httpx

    key = os.getenv("SERPAPI_API_KEY")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
        r = await http.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "num": max_results, "api_key": key},
        )
        r.raise_for_status()
        data = r.json()
    out: List[Dict[str, str]] = []
    for row in (data.get("organic_results") or []):
        url = (row.get("link") or "").strip()
        if not url:
            continue
        out.append({
            "title": (row.get("title") or "").strip() or url,
            "url": url,
            "snippet": (row.get("snippet") or "").strip(),
        })
    return out


async def _search_bing(query: str, max_results: int) -> List[Dict[str, str]]:
    import httpx

    key = os.getenv("BING_SEARCH_KEY")
    endpoint = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
        r = await http.get(
            endpoint,
            params={"q": query, "count": max_results},
            headers={"Ocp-Apim-Subscription-Key": key},
        )
        r.raise_for_status()
        data = r.json()
    out: List[Dict[str, str]] = []
    for row in ((data.get("webPages") or {}).get("value") or []):
        url = (row.get("url") or "").strip()
        if not url:
            continue
        out.append({
            "title": (row.get("name") or "").strip() or url,
            "url": url,
            "snippet": (row.get("snippet") or "").strip(),
        })
    return out


async def _search_google_cse(query: str, max_results: int) -> List[Dict[str, str]]:
    import httpx

    key = os.getenv("GOOGLE_CSE_KEY") or os.getenv("GOOGLE_CSE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as http:
        r = await http.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": query, "num": min(max_results, 10)},
        )
        r.raise_for_status()
        data = r.json()
    out: List[Dict[str, str]] = []
    for row in (data.get("items") or []):
        url = (row.get("link") or "").strip()
        if not url:
            continue
        out.append({
            "title": (row.get("title") or "").strip() or url,
            "url": url,
            "snippet": (row.get("snippet") or "").strip(),
        })
    return out


# ── Provider 3: degraded fallback (legacy WebSearchClient) ───────────────────
async def _search_degraded(query: str) -> List[Dict[str, str]]:
    """DEGRADED: wrap the legacy HTML-blob searcher.

    Returns a single item with an empty url; callers get at least the text.
    """
    try:
        from framework.Searchers.SearchInternet import WebSearchClient
    except Exception:
        return []
    buscador = WebSearchClient()
    text = await buscador.search_and_answer_web(query)
    text = _strip_html(text or "").strip()
    if not text:
        return []
    return [{"title": f"Search result: {query}"[:120], "url": "", "snippet": text}]


# ── Shared helpers ────────────────────────────────────────────────────────────
def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    try:
        return dict(obj)
    except Exception:
        return {
            "type": getattr(obj, "type", None),
            "url": getattr(obj, "url", None),
            "title": getattr(obj, "title", None),
            "start_index": getattr(obj, "start_index", None),
            "end_index": getattr(obj, "end_index", None),
        }


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _normalize(raw: List[Dict[str, str]], max_results: int) -> List[Dict[str, str]]:
    """Dedup by url, trim snippets, enforce shape and max_results."""
    out: List[Dict[str, str]] = []
    seen_urls = set()
    for row in (raw or []):
        if not isinstance(row, dict):
            continue
        url = (row.get("url") or "").strip()
        title = (row.get("title") or "").strip() or (url or "Source")
        snippet = (row.get("snippet") or "").strip()[:_SNIPPET_MAX]
        # Dedup only by non-empty urls; empty-url items (degraded) are kept.
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        out.append({"title": title, "url": url, "snippet": snippet})
        if len(out) >= max_results:
            break
    return out
