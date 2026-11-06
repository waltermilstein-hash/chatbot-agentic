# =============================================================================
# =============================================================================
"""
File generators for the "export tools" layer.

All generators:
  - Receive content (HTML or markdown for text; tabular data for xlsx;
    an outline for pptx) + a title.
  - Return a ``(bytes, filename)`` tuple — raw bytes ready to serve and a
    default filename with the correct extension.
  - Are DEFENSIVE: on empty/invalid content they generate a minimal valid
    file instead of raising.

Libraries used (already installed): python-docx, reportlab, openpyxl, python-pptx,
markdown, html.parser (stdlib).
"""

from __future__ import annotations

import io
import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


MIME_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "md": "text/markdown",
}

EXTENSIONS = {
    "docx": ".docx",
    "pdf": ".pdf",
    "xlsx": ".xlsx",
    "pptx": ".pptx",
    "md": ".md",
}


# ─────────────────────────── common utilities ──────────────────────────────

def _slugify_filename(title: str, ext: str) -> str:
    """Safe filename built from the title + extension."""
    base = (title or "document").strip() or "document"
    base = re.sub(r"[^\w\s.-]", "", base, flags=re.UNICODE)
    base = re.sub(r"\s+", "_", base).strip("_.") or "document"
    base = base[:80]
    if not ext.startswith("."):
        ext = "." + ext
    return f"{base}{ext}"


def _looks_like_html(content: str) -> bool:
    return bool(re.search(r"<[a-zA-Z/][^>]*>", content or ""))


# Intermediate block model: heading / para / bullet, each with "runs"
# [(text, bold, italic), ...]. Unifies HTML and markdown (markdown → HTML → parse).

class _DocModelParser(HTMLParser):
    _HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _SKIP = {"script", "style", "head", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: List[Dict[str, Any]] = []
        self._cur: Dict[str, Any] | None = None
        self._bold = 0
        self._italic = 0
        self._list_stack: List[str] = []
        self._skip = 0

    def _flush(self):
        if self._cur and any(t.strip() for t, _, _ in self._cur["runs"]):
            self.blocks.append(self._cur)
        self._cur = None

    def _start(self, kind: str, **kw):
        self._flush()
        self._cur = {"kind": kind, "runs": [], **kw}

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip += 1
            return
        if tag in self._HEADINGS:
            self._start("heading", level=int(tag[1]))
        elif tag in ("p", "div"):
            if tag == "div" and self._cur is not None:
                return
            self._start("para")
        elif tag in ("ul", "ol"):
            self._list_stack.append(tag)
        elif tag == "li":
            ordered = (self._list_stack[-1] == "ol") if self._list_stack else False
            self._start("bullet", ordered=ordered, level=max(1, len(self._list_stack)))
        elif tag in ("strong", "b"):
            self._bold += 1
        elif tag in ("em", "i"):
            self._italic += 1
        elif tag == "br":
            if self._cur is not None:
                self._cur["runs"].append(("\n", bool(self._bold), bool(self._italic)))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in self._HEADINGS or tag in ("p", "li"):
            self._flush()
        elif tag in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
        elif tag in ("strong", "b"):
            self._bold = max(0, self._bold - 1)
        elif tag in ("em", "i"):
            self._italic = max(0, self._italic - 1)

    def handle_data(self, data):
        if self._skip or not data:
            return
        text = re.sub(r"[ \t\r\n]+", " ", data)
        if not text.strip() and self._cur is None:
            return
        if self._cur is None:
            self._start("para")
        self._cur["runs"].append((text, bool(self._bold), bool(self._italic)))

    def close(self):  # noqa: A003
        super().close()
        self._flush()


def _parse_blocks(content: str) -> List[Dict[str, Any]]:
    """Convert HTML or markdown into a list of structured blocks."""
    content = content or ""
    if not _looks_like_html(content):
        try:
            import markdown as _md
            content = _md.markdown(content, extensions=["extra", "sane_lists"])
        except Exception:  # noqa: BLE001
            # No markdown: treat each non-empty line as a paragraph.
            return [
                {"kind": "para", "runs": [(ln.strip(), False, False)]}
                for ln in content.splitlines()
                if ln.strip()
            ]
    parser = _DocModelParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("export: HTML parse failed (%s); falling back to plain text", exc)
        plain = re.sub(r"<[^>]+>", " ", content)
        return [
            {"kind": "para", "runs": [(ln.strip(), False, False)]}
            for ln in plain.splitlines()
            if ln.strip()
        ]
    if not parser.blocks:
        plain = re.sub(r"<[^>]+>", " ", content).strip()
        if plain:
            return [{"kind": "para", "runs": [(plain, False, False)]}]
    return parser.blocks


def _runs_to_text(runs) -> str:
    return "".join(t for t, _, _ in runs).strip()


# ───────────────────────────── tables (xlsx) ───────────────────────────────

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[str]]] = []
        self._rows: List[List[str]] | None = None
        self._row: List[str] | None = None
        self._cell: List[str] | None = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table" and self._rows is not None:
            if self._rows:
                self.tables.append(self._rows)
            self._rows = None
        elif tag == "tr" and self._row is not None:
            self._rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _extract_html_tables(content: str) -> List[List[List[str]]]:
    p = _TableParser()
    try:
        p.feed(content)
        p.close()
    except Exception:  # noqa: BLE001
        return []
    return p.tables


# ─────────────────────────────── DOCX ──────────────────────────────────────

def to_docx(content: str, title: str = "Document") -> Tuple[bytes, str]:
    """HTML/markdown → clean .docx. Returns (bytes, filename)."""
    from docx import Document

    doc = Document()
    if title:
        doc.add_heading(str(title), level=0)

    try:
        blocks = _parse_blocks(content or "")
        if not blocks:
            doc.add_paragraph(re.sub(r"<[^>]+>", " ", content or "").strip())
        for blk in blocks:
            kind = blk.get("kind")
            if kind == "heading":
                lvl = min(max(int(blk.get("level", 2)), 1), 6)
                p = doc.add_heading("", level=lvl)
                _add_runs_docx(p, blk["runs"])
            elif kind == "bullet":
                style = "List Number" if blk.get("ordered") else "List Bullet"
                try:
                    p = doc.add_paragraph(style=style)
                except Exception:  # noqa: BLE001
                    p = doc.add_paragraph()
                _add_runs_docx(p, blk["runs"])
            else:
                p = doc.add_paragraph()
                _add_runs_docx(p, blk["runs"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("to_docx: render failed (%s); dumping plain text", exc)
        doc.add_paragraph(re.sub(r"<[^>]+>", " ", content or "").strip())

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue(), _slugify_filename(title, "docx")


def _add_runs_docx(paragraph, runs):
    for text, bold, italic in runs:
        for i, part in enumerate(text.split("\n")):
            if i > 0:
                paragraph.add_run().add_break()
            if part:
                r = paragraph.add_run(part)
                r.bold = bool(bold)
                r.italic = bool(italic)


# ─────────────────────────────── PDF ───────────────────────────────────────

def to_pdf(content: str, title: str = "Document") -> Tuple[bytes, str]:
    """HTML/markdown → readable PDF via reportlab (platypus). Returns (bytes, filename)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=str(title or "Document"),
    )
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.spaceAfter = 6
    h_styles = {
        1: styles["Heading1"], 2: styles["Heading2"], 3: styles["Heading3"],
        4: styles["Heading4"], 5: styles["Heading5"], 6: styles["Heading6"],
    }
    title_style = ParagraphStyle("DocTitle", parent=styles["Title"], spaceAfter=14)

    story: List[Any] = []
    if title:
        story.append(Paragraph(_escape_rl(str(title)), title_style))

    try:
        blocks = _parse_blocks(content or "")
        if not blocks:
            story.append(Paragraph(_escape_rl(re.sub(r"<[^>]+>", " ", content or "").strip()) or "&nbsp;", body))
        pending_bullets: List[Any] = []

        def _flush_bullets():
            if pending_bullets:
                story.append(ListFlowable(list(pending_bullets), bulletType="bullet", leftIndent=18))
                story.append(Spacer(1, 4))
                pending_bullets.clear()

        for blk in blocks:
            kind = blk.get("kind")
            markup = _runs_to_rl_markup(blk["runs"])
            if not markup.strip():
                continue
            if kind == "bullet":
                pending_bullets.append(ListItem(Paragraph(markup, body), leftIndent=12))
                continue
            _flush_bullets()
            if kind == "heading":
                lvl = min(max(int(blk.get("level", 2)), 1), 6)
                story.append(Paragraph(markup, h_styles[lvl]))
            else:
                story.append(Paragraph(markup, body))
        _flush_bullets()
    except Exception as exc:  # noqa: BLE001
        logger.warning("to_pdf: render failed (%s); dumping plain text", exc)
        story = [Paragraph(_escape_rl(re.sub(r"<[^>]+>", " ", content or "").strip()) or "&nbsp;", body)]

    if not story:
        story = [Paragraph("&nbsp;", body)]
    doc.build(story)
    return buf.getvalue(), _slugify_filename(title, "pdf")


def _escape_rl(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _runs_to_rl_markup(runs) -> str:
    out = []
    for text, bold, italic in runs:
        chunk = _escape_rl(text).replace("\n", "<br/>")
        if bold:
            chunk = f"<b>{chunk}</b>"
        if italic:
            chunk = f"<i>{chunk}</i>"
        out.append(chunk)
    return "".join(out)


# ─────────────────────────────── XLSX ──────────────────────────────────────

def to_xlsx(data: Any, title: str = "Data") -> Tuple[bytes, str]:
    """
    data may be:
      - list of dicts  → columns = keys
      - list of lists  → direct rows
      - HTML string    → the first table is extracted
      - plain string   → one cell per line
    If there is nothing tabular, it dumps the text into a sheet (never breaks).
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = (str(title) or "Data")[:31] or "Data"

    rows, header = _coerce_rows(data)
    try:
        if header:
            ws.append(header)
            for c in ws[1]:
                c.font = Font(bold=True)
        for row in rows:
            ws.append(row)
        if not header and not rows:
            ws.append(["(no data)"])
        _autosize(ws)
    except Exception as exc:  # noqa: BLE001
        logger.warning("to_xlsx: failed (%s); dumping text", exc)
        ws = wb.active
        ws.append([str(data)[:3000]])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue(), _slugify_filename(title, "xlsx")


def _coerce_rows(data: Any):
    """Return (rows, header) from heterogeneous data."""
    # HTML with a table
    if isinstance(data, str) and _looks_like_html(data):
        tables = _extract_html_tables(data)
        if tables:
            t = tables[0]
            if len(t) >= 2:
                return [list(r) for r in t[1:]], list(t[0])
            return [list(r) for r in t], None
        plain = re.sub(r"<[^>]+>", " ", data)
        return [[ln.strip()] for ln in plain.splitlines() if ln.strip()], None
    # plain text
    if isinstance(data, str):
        return [[ln] for ln in data.splitlines() if ln.strip()], None
    # list
    if isinstance(data, (list, tuple)):
        items = list(data)
        if not items:
            return [], None
        if all(isinstance(it, dict) for it in items):
            header: List[str] = []
            for it in items:
                for k in it.keys():
                    if k not in header:
                        header.append(str(k))
            rows = [[_cell(it.get(k)) for k in header] for it in items]
            return rows, header
        if all(isinstance(it, (list, tuple)) for it in items):
            return [[_cell(c) for c in it] for it in items], None
        return [[_cell(it)] for it in items], None
    # loose dict
    if isinstance(data, dict):
        return [[str(k), _cell(v)] for k, v in data.items()], ["Key", "Value"]
    return [[str(data)]], None


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _autosize(ws):
    try:
        for col in ws.columns:
            width = 10
            letter = col[0].column_letter
            for cell in col:
                val = cell.value
                if val is not None:
                    width = max(width, min(60, len(str(val)) + 2))
            ws.column_dimensions[letter].width = width
    except Exception:  # noqa: BLE001
        pass


# ─────────────────────────────── PPTX ──────────────────────────────────────

def to_pptx(outline: Dict[str, Any], title: str = "Presentation") -> Tuple[bytes, str]:
    """
    outline = {"title": str, "slides": [{"title": str, "bullets": [str, ...]}, ...]}
    Generates a .pptx with a cover slide + one slide per item. Defensive against
    malformed outlines.
    """
    from pptx import Presentation
    from pptx.util import Pt

    outline = outline if isinstance(outline, dict) else {}
    deck_title = str(outline.get("title") or title or "Presentation")
    slides = outline.get("slides")
    if not isinstance(slides, list):
        slides = []

    prs = Presentation()

    # Cover
    try:
        cover = prs.slides.add_slide(prs.slide_layouts[0])
        cover.shapes.title.text = deck_title
        if len(cover.placeholders) > 1:
            cover.placeholders[1].text = "Generated"
    except Exception:  # noqa: BLE001
        pass

    content_layout = prs.slide_layouts[1]
    for sl in slides:
        if not isinstance(sl, dict):
            continue
        try:
            slide = prs.slides.add_slide(content_layout)
            slide.shapes.title.text = str(sl.get("title") or "")
            bullets = sl.get("bullets") or []
            if not isinstance(bullets, list):
                bullets = [str(bullets)]
            body = None
            for ph in slide.placeholders:
                if ph.placeholder_format.idx == 1:
                    body = ph
                    break
            if body is None:
                continue
            tf = body.text_frame
            tf.clear()
            first = True
            for b in bullets:
                text = str(b).strip()
                if not text:
                    continue
                p = tf.paragraphs[0] if first else tf.add_paragraph()
                p.text = text
                p.level = 0
                p.font.size = Pt(18)
                first = False
            if first:  # no bullets
                tf.paragraphs[0].text = ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("to_pptx: slide skipped (%s)", exc)
            continue

    # Ensure at least one content slide if there were none.
    if len(prs.slides) <= 1 and not slides:
        try:
            s = prs.slides.add_slide(content_layout)
            s.shapes.title.text = deck_title
        except Exception:  # noqa: BLE001
            pass

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue(), _slugify_filename(deck_title or title, "pptx")


# ─────────────────────────── Markdown (.md) ─────────────────────────────────
def to_md(content: str, title: str = "Document"):
    """Export to Markdown. Converts HTML→basic markdown if the content is HTML."""
    import html as _html
    import re as _re
    txt = content or ""
    if _looks_like_html(txt):
        txt = _re.sub(r'(?is)<h[12][^>]*>(.*?)</h[12]>', r'\n## \1\n', txt)
        txt = _re.sub(r'(?is)<h[3-6][^>]*>(.*?)</h[3-6]>', r'\n### \1\n', txt)
        txt = _re.sub(r'(?is)<(strong|b)>(.*?)</\1>', r'**\2**', txt)
        txt = _re.sub(r'(?is)<(em|i)>(.*?)</\1>', r'*\2*', txt)
        txt = _re.sub(r'(?is)<li[^>]*>(.*?)</li>', r'- \1\n', txt)
        txt = _re.sub(r'(?is)<(summary)[^>]*>(.*?)</summary>', r'\n**\2**\n', txt)
        txt = _re.sub(r'(?is)<(p|div|br|details|ul|ol)[^>]*>', '\n', txt)
        txt = _re.sub(r'(?is)</(p|div|details|ul|ol)>', '\n', txt)
        txt = _re.sub(r'(?is)<[^>]+>', '', txt)
        txt = _html.unescape(txt)
        txt = _re.sub(r'[ \t]+\n', '\n', txt)
        txt = _re.sub(r'\n{3,}', '\n\n', txt).strip()
    md = f"# {title}\n\n{txt}\n"
    return md.encode("utf-8"), _slugify_filename(title, ".md")
