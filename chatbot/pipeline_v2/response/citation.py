# =============================================================================
# =============================================================================
"""
CitationBuilder — Structured citation of document sources.

In a law firm, every statement must be traceable to its source document,
clause and page. This module extracts that information from the chunks
retrieved by the retriever and appends it to the HTML answer.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CitationBuilder:
    """Builds and appends source citations to the chatbot's answers."""

    def extract_citations(self, docs: List[Dict]) -> List[Dict]:
        """
        Extract citation metadata from the retrieved documents.

        Returns a list of:
        {
            "indice": int,
            "documento": str,
            "pagina": int | None,
            "clausula": str | None,
            "fragmento": str,
            "score": float,
        }
        (These keys are part of the SSE "sources" payload consumed by the frontend.)
        """
        citations = []
        seen = set()

        for i, doc in enumerate(docs):
            name = (
                doc.get("filename")
                or doc.get("nombre")
                or doc.get("source")
                or doc.get("fuente")
                or f"Document {i + 1}"
            )
            page = doc.get("pagina") or doc.get("page") or doc.get("page_number")
            clause = self._detect_clausula(doc)
            snippet = str(
                doc.get("content") or doc.get("contenido") or doc.get("text") or ""
            )[:300]
            score = float(doc.get("certainty") or doc.get("score") or doc.get("similitud") or 0.0)

            key = (name, page, clause)
            if key in seen:
                continue
            seen.add(key)

            citations.append({
                "indice": len(citations) + 1,
                "documento": name,
                "pagina": int(page) if page is not None else None,
                "clausula": clause,
                "fragmento": snippet,
                "score": round(score, 4),
            })

        return citations

    def append_citations_html(self, answer: str, citations: List[Dict]) -> str:
        """Append the document references block to the answer's HTML."""
        if not citations:
            return answer

        lines = [
            "<hr>",
            '<div class="citations">',
            "<h4>📚 Document sources</h4>",
            "<ol>",
        ]

        for c in citations:
            location = self._format_ubicacion(c)
            score_pct = f"{c['score'] * 100:.0f}%" if c["score"] > 0 else "—"
            lines.append(
                f'<li><strong>{c["document"]}</strong>'
                f"{location}"
                f'<br><em style="color:#777;font-size:0.9em">'
                f'Relevance: {score_pct} · «{c["fragmento"][:120]}…»'
                f"</em></li>"
            )

        lines += ["</ol>", "</div>"]
        return answer + "\n" + "\n".join(lines)

    def _detect_clausula(self, doc: Dict) -> Optional[str]:
        """Try to extract the clause/article number from the document."""
        # From explicit metadata
        for key in ("clausula", "clause", "articulo", "article", "section", "seccion"):
            if doc.get(key):
                return str(doc[key])

        # From the content via regex (matches Spanish legal references in the text)
        content = str(doc.get("content") or doc.get("contenido") or "")[:500]
        match = re.search(
            r"(cl[áa]usula|art[íi]culo|secci[óo]n|cap[íi]tulo|punto)\s+(\d+[\w\.]*)",
            content,
            re.IGNORECASE,
        )
        if match:
            return f"{match.group(1).capitalize()} {match.group(2)}"
        return None

    def _format_ubicacion(self, citation: Dict) -> str:
        parts = []
        if citation.get("pagina"):
            parts.append(f"p. {citation['page']}")
        if citation.get("clausula"):
            parts.append(citation["clausula"])
        return f" — {', '.join(parts)}" if parts else ""
