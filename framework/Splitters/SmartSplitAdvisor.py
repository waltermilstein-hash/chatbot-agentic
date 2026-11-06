"""
"""
# smart_split_advisor.py
from __future__ import annotations
import os, json, logging, re, statistics
from typing import Dict, Any, Optional

from config.Constants import GPT_5_1, GPT_5_NANO
# Keep this import as-is (your real path). If you use another, adjust externally.
from framework.Splitters.ParallelDocumentSplitter import SplitConfig

logger = logging.getLogger(__name__)

# =========================
#   Internal utilities
# =========================

# Short list of abbreviations to avoid false sentence breaks (Spanish + common)
_ABBREV = r"(?:Sr|Sra|Srta|Dr|Dra|Ing|Lic|Ph\.D|Ms|Mrs|Mr|etc|p\.ej|e\.g|i\.e)"

# Sentence splitter tolerant to Spanish (¿¡), quotes and parentheses.
# Avoids breaking after common abbreviations.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ\"'“”¿¡\(\[])"
)
def split_sentences(text: str) -> list[str]:
    # protect common abbreviations
    text = re.sub(r"\b(Sr|Sra|Sra|Dr|Dra|Ing|Lic|Ph\.D|etc|p\.ej|e\.g|i\.e)\.", r"\1§", text)
    parts = _SENT_SPLIT.split(text)
    # restore the period
    return [p.replace("§", ".").strip() for p in parts if p.strip()]


def _extract_json(s: str) -> Optional[dict]:
    """Try to extract the first valid JSON block, tolerating ```fences``` and simple trailing commas."""
    if not s:
        return None
    s = s.strip()
    # strip ```json ... ``` fences
    if s.startswith("```"):
        s = re.sub(r"^```\w*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # locate the first {...}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    blob = m.group(0)
    # remove simple trailing commas before a closing bracket
    blob = re.sub(r",\s*([\}\]])", r"\1", blob)
    try:
        return json.loads(blob)
    except Exception:
        return None

def _clamp_overlap(cfg: SplitConfig) -> SplitConfig:
    """Adjust overlap to avoid absurd values and ensure stride >= min_chunk_size."""
    max_overlap = int(cfg.chunk_size * 0.5)  # for safety, no more than 50%
    cfg.overlap = min(max(0, cfg.overlap), max_overlap)
    stride = cfg.chunk_size - cfg.overlap
    if stride < cfg.min_chunk_size:
        cfg.overlap = max(0, cfg.chunk_size - cfg.min_chunk_size)
    return cfg

def _looks_structured(text: str) -> bool:
    """Additional signals of structured content (tables, CSV-ish, high digit ratio)."""
    if not text:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    pipe_lines = sum(1 for l in lines if l.count("|") >= 2)  # markdown tables
    csvish = sum(1 for l in lines if (l.count(",") >= 3 or l.count(";") >= 3 or "\t" in l))
    digit_ratio = (sum(c.isdigit() for c in text) / max(1, len(text)))
    return (pipe_lines + csvish) > 5 or digit_ratio > 0.22

# =========================
#   Basic stats
# =========================

def _basic_stats(text: str) -> Dict[str, Any]:
    lines = [l for l in text.splitlines() if l.strip()]
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sent_counts = []
    for p in paras[:200]:
        sents = _SENT_SPLIT.split(p)
        sent_counts.append(len(sents))
    tables = sum(1 for l in lines if l.count("|") >= 2)  # more precise than just "|" in l
    code_fences = len(re.findall(r"(^```.*?^```)", text, flags=re.M | re.S))
    lists = sum(1 for l in lines if re.match(r"^\s*([\-*+]|[0-9]+\.)\s+", l))
    headings = sum(1 for l in lines if re.match(r"^\s*#{1,6}\s+\S", l))
    avg_para_len = statistics.mean([len(p) for p in paras[:500]]) if paras else 0
    median_para_len = statistics.median([len(p) for p in paras[:500]]) if paras else 0
    digit_ratio = (sum(c.isdigit() for c in text) / max(1, len(text))) if text else 0.0
    return {
        "lines": len(lines),
        "paras": len(paras),
        "avg_para_len": avg_para_len,
        "median_para_len": median_para_len,
        "tables": tables,
        "code_fences": code_fences,
        "lists": lists,
        "headings": headings,
        "digit_ratio": round(digit_ratio, 4),
        "long_doc": len(text) > 100_000
    }

# =========================
#   Main heuristic
# =========================

def _heuristic_config(text: str) -> SplitConfig:
    """
    Keeps the original signature. Reinforced heuristic for RAG with structure detection.
    """
    n = len(text)
    # Look at a large slice for speed (and to avoid bias from an isolated page)
    probe = text[:20000]
    stats = _basic_stats(probe)
    content_is_structured = ((stats["tables"] + stats["code_fences"] + stats["lists"]) > 3) or _looks_structured(probe)

    # Base by length - OPTIMIZED FOR RAG
    if n < 5_000:
        base = 300
    elif n < 50_000:
        base = 500
    elif n < 200_000:
        base = 800
    else:
        base = 1200

    # Structure adjustments
    if content_is_structured:
        # smaller chunks for tables/lists/csv-ish
        base = max(200, int(base * 0.6))
    if stats["median_para_len"] > 1000:
        base = int(base * 1.1)  # giant paragraphs -> a bit larger

    # Generous overlap for RAG (≈25%)
    overlap = max(50, int(base * 0.25))

    # For structured content, don't preserve paragraphs to avoid gluing heterogeneous blocks
    preserve_paragraphs = not content_is_structured

    cfg = SplitConfig(
        chunk_size=base,
        overlap=overlap,
        preserve_paragraphs=preserve_paragraphs,
        min_chunk_size=100  # filters junk/noise
    )
    cfg = _clamp_overlap(cfg)

    stride = cfg.chunk_size - cfg.overlap
    logger.info(f"[HEURISTIC] n={n}, structured={content_is_structured}, "
                f"chunk={cfg.chunk_size}, overlap={cfg.overlap}, stride={stride}, "
                f"preserve_paragraphs={preserve_paragraphs}")
    return cfg

# =========================
#   LLM advisor (optional)
# =========================

class SmartSplitAdvisor:
    """
    Recommends a SplitConfig using an LLM (optional) + a RAG-optimized heuristic.
    Keeps the same public signatures.
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None, llm_fn=None):
        self.provider = provider
        self.model =GPT_5_NANO
        self.llm_fn = llm_fn
        self.max_preview_chars = 24_000
        self.cost_guard_chars = 120_000

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            if self.llm_fn:
                return self.llm_fn(prompt)

            if self.provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set; using heuristic")
                    return None
                from openai import OpenAI
                client = OpenAI(api_key=api_key, max_retries=0, timeout=60.0)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]

                )
                return resp.choices[0].message.content

            if self.provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    logger.warning("GEMINI_API_KEY not set; using heuristic")
                    return None
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content(prompt)
                return getattr(resp, "text", None)

            logger.warning("Unsupported LLM provider; using heuristic")
            return None
        except Exception as e:
            logger.error(f"Error calling the LLM: {e}")
            return None

    def _tri_sample(self,text: str, per_slice: int = 1000) -> str:
        """
        Return 3 representative slices of the document: start, middle and end.
        Useful so the LLM sees diversity, not just the first page.
        """
        n = len(text)
        if n <= per_slice * 3:
            return text[:per_slice * 3]
        head = text[:per_slice]
        mid_start = max(0, n // 2 - per_slice // 2)
        middle = text[mid_start: mid_start + per_slice]
        tail = text[-per_slice:]
        return f"<<<HEAD>>>\n{head}\n<<<MIDDLE>>>\n{middle}\n<<<TAIL>>>\n{tail}"

    def advise_config(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> SplitConfig:
        """
        Return a SplitConfig optimized for generic RAG.
        """
        if not text or not text.strip():
            return SplitConfig()

        text_len = len(text)

        # Representative sample: head/middle/tail within the preview cap
        raw_preview = text[:self.max_preview_chars]
        preview = self._tri_sample(raw_preview, per_slice=1000)
        stats = _basic_stats(preview)
        preview_lower = preview.lower()

        # === AUTOMATIC DOCUMENT-TYPE DETECTION (matches Spanish document content) ===
        is_legal = bool(re.search(
            r'\b(artículo|art\.|sección|sec\.|capítulo|cap\.|convenio|ley|decreto|disposición)\b',
            preview_lower
        ))
        is_technical = bool(re.search(
            r'\b(tabla|figura|anexo|apéndice|ISO[-\s]?\d+|CNAE|sección\s+\d+)\b',
            preview_lower
        ))
        has_many_refs = len(re.findall(
            r'\b\w*\d+[a-z]+\b|\b[a-z]+\d+\b',
            preview_lower
        )) > 5
        digit_density = stats.get('digit_density', 0)

        # === ADAPTIVE CONFIGURATION ===
        if is_legal:
            default_chunk = 700
            default_overlap_pct = 0.30
            default_preserve = True
            doc_type = "legal/regulatory"
        elif is_technical or has_many_refs:
            default_chunk = 650
            default_overlap_pct = 0.28
            default_preserve = True
            doc_type = "technical"
        elif digit_density > 0.15:
            default_chunk = 500
            default_overlap_pct = 0.25
            default_preserve = False
            doc_type = "structured/tabular"
        else:
            default_chunk = 800
            default_overlap_pct = 0.20
            default_preserve = True
            doc_type = "narrative"

        # ✅ GUARD: short text → no point splitting
        # Evaluated AFTER detecting the type so the correct default_chunk is used
        if text_len <= default_chunk:
            logger.info(
                f"[SHORT_TEXT] type={doc_type}, text_len={text_len} <= chunk={default_chunk}; skip split"
            )
            return SplitConfig(
                chunk_size=text_len,
                overlap=0,
                preserve_paragraphs=default_preserve,
                min_chunk_size=text_len,
            )

        # === LLM PROMPT ===
        prompt = f"""
    You are an expert in chunking for RAG. Return ONLY valid JSON with the fields:
    - chunk_size: int (300..1200)
    - overlap: int (100..chunk_size/2)
    - preserve_paragraphs: bool
    - min_chunk_size: int (>= 80)

    DETECTED DOCUMENT TYPE: {doc_type}

    TYPE-SPECIFIC RULES:

    1) LEGAL/REGULATORY DOCUMENTS:
       - chunk_size: 650-850 | overlap: 28-32% | preserve_paragraphs: true

    2) TECHNICAL DOCUMENTS:
       - chunk_size: 600-800 | overlap: 25-30% | preserve_paragraphs: true

    3) STRUCTURED/TABULAR:
       - chunk_size: 400-600 | overlap: 20-25% | preserve_paragraphs: false

    4) GENERAL NARRATIVE:
       - chunk_size: 700-1000 | overlap: 18-22% | preserve_paragraphs: true

    HARD RULES:
    - MINIMUM overlap: 100 | MAXIMUM chunk_size: 1200
    - stride = chunk_size - overlap >= min_chunk_size

    DOCUMENT ANALYSIS:
    - Type: {doc_type} | Is legal: {is_legal} | Is technical: {is_technical}
    - Has many refs: {has_many_refs} | Digit density: {digit_density:.1%}
    - Total length: {text_len} chars

    SUGGESTED DEFAULTS:
    - chunk_size: {default_chunk}
    - overlap: {int(default_chunk * default_overlap_pct)} ({default_overlap_pct:.0%})
    - preserve_paragraphs: {str(default_preserve).lower()}

    SAMPLE (head/middle/tail):
    {preview[:1500]}

    OUTPUT (strict JSON):
    {{
      "chunk_size": <int 300..1200>,
      "overlap": <int 100..chunk_size/2>,
      "preserve_paragraphs": <true|false>,
      "min_chunk_size": <int >= 80>
    }}
    """.strip()

        # === LLM CALL (only if the doc is not huge) ===
        if text_len < self.cost_guard_chars:
            llm_answer = self._call_llm(prompt)
            if llm_answer:
                data = _extract_json(llm_answer)
                if data:
                    chunk_size = int(data.get("chunk_size", default_chunk))
                    overlap = int(data.get("overlap", int(default_chunk * default_overlap_pct)))
                    preserve_paragraphs = bool(data.get("preserve_paragraphs", default_preserve))
                    min_chunk_size = int(data.get("min_chunk_size", 100))

                    # === VALIDATIONS ===
                    chunk_size = max(300, min(1200, chunk_size))
                    overlap = max(100, min(overlap, chunk_size // 2))
                    min_chunk_size = max(80, min_chunk_size)

                    stride = chunk_size - overlap
                    if stride < min_chunk_size:
                        overlap = chunk_size - min_chunk_size
                        logger.warning(f"Adjusting overlap to {overlap} for stride >= min_chunk_size")

                    cfg = SplitConfig(
                        chunk_size=chunk_size,
                        overlap=overlap,
                        preserve_paragraphs=preserve_paragraphs,
                        min_chunk_size=min_chunk_size,
                    )
                    cfg = _clamp_overlap(cfg)

                    stride = cfg.chunk_size - cfg.overlap
                    overlap_pct = (cfg.overlap / cfg.chunk_size) * 100
                    logger.info(
                        f"[LLM] type={doc_type}, chunk={cfg.chunk_size}, "
                        f"overlap={cfg.overlap} ({overlap_pct:.0f}%), stride={stride}, "
                        f"preserve={cfg.preserve_paragraphs}, min_chunk={cfg.min_chunk_size}"
                    )
                    return cfg
                else:
                    logger.error("[LLM] Could not extract valid JSON; using heuristic")

        # === HEURISTIC FALLBACK ===
        fallback_cfg = SplitConfig(
            chunk_size=default_chunk,
            overlap=max(100, int(default_chunk * default_overlap_pct)),
            preserve_paragraphs=default_preserve,
            min_chunk_size=100,
        )
        logger.info(
            f"[FALLBACK] type={doc_type}, chunk={fallback_cfg.chunk_size}, "
            f"overlap={fallback_cfg.overlap} ({default_overlap_pct:.0%})"
        )
        return fallback_cfg
