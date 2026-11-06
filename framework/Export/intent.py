# =============================================================================
# =============================================================================
"""
Export intent detection ("export tools" layer).

`detect_export(message) -> {"format": "docx"|"pdf"|"pptx"|"xlsx"|None, "content_hint": str}`

REGEX-based approach (ES + EN), NOT function-calling. Designed to be precise: a
normal documental query (e.g. "What does the contract PDF say?") must NOT trigger
an export. Besides a format token, an explicit intent signal is required:
  - an export verb (ponelo/pasá/convertí/exportá/generá/creá/hacé/…,
    make/create/export/convert/turn into/put in/give me/save as), or
  - the word "descargable"/"para descargar"/"downloadable", or
  - the message STARTING with the format token (imperative style:
    "PDF con el resumen", "Excel de los contratos").
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# ── Format tokens → canonical format ────────────────────────────────────────
_FORMAT_PATTERNS = [
    ("pptx", r"\b(pptx|powerpoint|power\s?point|ppt|presentaci[oó]n|presentacion|diapositivas?|slides?)\b|\.pptx\b"),
    ("xlsx", r"\b(excel|xlsx|planilla|spreadsheet|hoja\s+de\s+c[aá]lculo)\b|\.xlsx\b"),
    ("docx", r"\b(word|docx|documento\s+word|ms\s?word|microsoft\s+word)\b|\.docx\b"),
    ("pdf", r"\b(pdf)\b|\.pdf\b"),
    ("md", r"\b(markdown|md)\b|\.md\b"),
]

# ── Export verbs / triggers ─────────────────────────────────────────────────
_VERB_RE = re.compile(
    r"\b("
    # Spanish (imperative voseo/tuteo + first person) — matches Spanish user input
    r"pon[eé]lo|p[oó]nlo|pon[eé]r?|ponme|pas[aá]lo|p[aá]salo|pas[aá](me|melo)?|p[aá]sa(me|melo)?|"
    r"convert[ií](lo|me|melo|r)?|convierte(me|lo)?|transform[aá]r?|"
    r"export[aá]r?|exp[oó]rtalo|export[aá]me|"
    r"gener[aá]r?|gen[eé]ralo|gener[aá]me|"
    r"cre[aá]r?|cr[eé]alo|cre[aá]me|"
    r"hac[eé]r?|hac[eé]me|hac[eé]lo|haz(me|lo)?|h[aá]game|"
    r"arm[aá]r?|[aá]rmalo|arm[aá]me|"
    r"descarg[aá]r?|b[aá]jame|baj[aá](me)?|"
    r"dame|quiero|necesito|me\s+gustar[ií]a|"
    # English
    r"make|create|generate|export|convert|download|save\s+as|"
    r"turn\s+(this|it|that)?\s*into|put\s+(this|it|that)?\s*(in|into)|give\s+me|i\s+(want|need)"
    r")\b",
    re.IGNORECASE,
)

_DESCARGABLE_RE = re.compile(
    r"\b(descargable|para\s+descargar|downloadable|as\s+a\s+file)\b", re.IGNORECASE
)

# Content hints: "con el resumen", "de los contratos", "con el resultado",
# "de esto", "sobre X", "with the summary", "of the contracts"…
_HINT_RE = re.compile(
    r"\b(?:con|de|del|sobre|about|with|of|for)\s+"
    r"(?:el|la|los|las|lo|un|una|unos|unas|mi|mis|tu|sus|este|esta|esto|eso|esa|ese|estos|estas|"
    r"the|this|that|these|those|a|an|my|your)?\s*"
    r"(.{2,80}?)\s*$",
    re.IGNORECASE,
)

# Words that imply generating NEW CONTENT (not exporting the last answer as-is).
_NEW_CONTENT_RE = re.compile(
    r"\b(resumen|res[uú]men|resum[ií]|resume|resumen\s+ejecutivo|"
    r"comparativ[oa]|comparaci[oó]n|comparar|compar[aá]|tabla\s+comparativa|"
    r"an[aá]lisis|analiz[aá]|informe|reporte|report|outline|esquema|"
    r"summary|comparison|compare|analysis|overview)\b",
    re.IGNORECASE,
)


def _detect_format(msg: str) -> Optional[str]:
    for fmt, pattern in _FORMAT_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return fmt
    return None


def _leads_with_format(msg: str) -> bool:
    """Does the message start (imperative style) with a format token?"""
    lead = re.match(
        r"^[\s¿¡\-\*>\"']*(?:un\s+|una\s+|el\s+|la\s+|a\s+|an\s+)?"
        r"(word|pdf|pptx|powerpoint|power\s?point|ppt|excel|xlsx|planilla|markdown|md|"
        r"presentaci[oó]n|presentacion|documento\s+word|slides?|diapositivas?)\b",
        msg, re.IGNORECASE,
    )
    return bool(lead)


def _extract_hint(msg: str) -> str:
    m = _HINT_RE.search(msg.strip())
    if not m:
        return ""
    hint = (m.group(1) or "").strip(" .,:;!?\"'")
    # Avoid capturing only the format name.
    if re.fullmatch(r"(word|pdf|pptx|powerpoint|ppt|excel|xlsx|planilla|archivo|file|documento)\.?",
                    hint, re.IGNORECASE):
        return ""
    return hint


def detect_export(message: str) -> Dict[str, object]:
    """Return {"format": <fmt|None>, "content_hint": str}."""
    msg = (message or "").strip()
    if not msg:
        return {"format": None, "content_hint": ""}

    fmt = _detect_format(msg)
    if not fmt:
        return {"format": None, "content_hint": ""}

    has_verb = bool(_VERB_RE.search(msg))
    has_descargable = bool(_DESCARGABLE_RE.search(msg))
    leads = _leads_with_format(msg)

    if not (has_verb or has_descargable or leads):
        return {"format": None, "content_hint": ""}

    return {"format": fmt, "content_hint": _extract_hint(msg)}


def wants_new_content(text: str) -> bool:
    """True if the hint/message asks to generate NEW content (summary, comparison…)."""
    return bool(_NEW_CONTENT_RE.search(text or ""))
