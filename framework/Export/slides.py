# =============================================================================
# =============================================================================
"""
Build presentation outlines with an LLM ("export tools" layer).

`build_slide_outline(content, title, llm=None) -> dict`

Converts long content (HTML/markdown/text) into a JSON outline:
    {"title": str, "slides": [{"title": str, "bullets": [str, ...]}, ...]}

Uses the project's LLM (OpenAIProvider / gpt-5.4-mini by default), reusing the
`chatbot.pipeline_v2.llm_provider` abstraction. Best-effort: on any failure
(no network, invalid JSON, etc.) it falls back to a naive split that NEVER
breaks.

Optional `llm`: async callable `(prompt: str) -> str`. If omitted, the project's
OpenAI provider is used.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_MIN_SLIDES = 6
_MAX_SLIDES = 10

_PROMPT = (
    "Turn the following content into the outline of a professional presentation.\n"
    "Return EXCLUSIVELY valid JSON with this EXACT shape, with no extra text "
    "or markdown:\n"
    '{{"title": "<presentation title>", "slides": [{{"title": "<slide title>", '
    '"bullets": ["<concise bullet>", "..."]}}]}}\n'
    "Rules: between {min_s} and {max_s} slides; 3 to 5 bullets per slide; short "
    "bullets (max ~15 words), clear and without HTML. Language: the same as the content.\n\n"
    "SUGGESTED TITLE: {title}\n\n"
    "CONTENT:\n{content}"
)


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text or "", flags=re.IGNORECASE)
    text = re.sub(r"</(p|li|h[1-6]|div|tr)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    return None


def _normalize_outline(obj: Any, title: str) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    slides_raw = obj.get("slides")
    if not isinstance(slides_raw, list) or not slides_raw:
        return None
    slides: List[Dict[str, Any]] = []
    for sl in slides_raw[:_MAX_SLIDES]:
        if not isinstance(sl, dict):
            continue
        st = str(sl.get("title") or "").strip()
        bullets_raw = sl.get("bullets") or []
        if not isinstance(bullets_raw, list):
            bullets_raw = [str(bullets_raw)]
        bullets = [str(b).strip() for b in bullets_raw if str(b).strip()][:6]
        if st or bullets:
            slides.append({"title": st or title, "bullets": bullets})
    if not slides:
        return None
    return {"title": str(obj.get("title") or title or "Presentation"), "slides": slides}


def _naive_outline(content: str, title: str) -> dict:
    """LLM-free fallback: split the text into blocks and build slides with bullets."""
    plain = _strip_html(content)
    # Prefer explicit lines/bullets; otherwise, sentences.
    lines = [ln.strip(" -*•\t") for ln in plain.splitlines() if ln.strip(" -*•\t")]
    if len(lines) < _MIN_SLIDES:
        sentences = re.split(r"(?<=[.!?])\s+", plain)
        lines = [s.strip() for s in sentences if len(s.strip()) > 2] or lines
    if not lines:
        lines = [title or "Content"]

    bullets_per_slide = 4
    chunks = [lines[i:i + bullets_per_slide] for i in range(0, len(lines), bullets_per_slide)]
    chunks = chunks[:_MAX_SLIDES] or [[title or "Content"]]

    slides = []
    for i, ch in enumerate(chunks, 1):
        head = ch[0][:60] if ch else f"Section {i}"
        slides.append({
            "title": head if len(ch) == 1 else f"Section {i}",
            "bullets": [b[:160] for b in ch],
        })
    return {"title": title or "Presentation", "slides": slides}


async def build_slide_outline(
    content: str,
    title: str = "Presentation",
    llm: Optional[Callable[[str], Any]] = None,
) -> dict:
    """
    Generate the outline with the project's LLM. Best-effort → naive fallback.

    Args:
        content: source text/HTML/markdown.
        title:   suggested title.
        llm:     optional async callable `(prompt) -> str`. If None, uses OpenAIProvider.
    """
    source = _strip_html(content or "")
    if not source.strip():
        return _naive_outline(content or (title or ""), title)

    prompt = _PROMPT.format(
        min_s=_MIN_SLIDES, max_s=_MAX_SLIDES,
        title=title or "Presentation", content=source[:8000],
    )

    raw = None
    try:
        if llm is not None:
            res = llm(prompt)
            if hasattr(res, "__await__"):
                res = await res
            raw = str(res)
        else:
            from chatbot.pipeline_v2.llm_provider import get_provider
            from config import Constants
            provider = get_provider("openai")
            model = getattr(Constants, "GPT_5_4_MINI", "gpt-5.4-mini")
            raw = await provider.complete(
                [
                    {"role": "system", "content": "You are an assistant that returns ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model, temperature=0.3, max_tokens=1500,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_slide_outline: LLM failed (%s); naive fallback", exc)
        return _naive_outline(content, title)

    outline = _normalize_outline(_extract_json(raw), title)
    if outline:
        return outline
    logger.info("build_slide_outline: invalid LLM outline; naive fallback")
    return _naive_outline(content, title)
