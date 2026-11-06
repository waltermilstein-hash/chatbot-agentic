"""
"""
import json
import os
import time

from config.Constants import KEY_OPEN_AI, GPT_5_NANO, GPT_5_1, GPT_5_4

os.environ['OPENAI_API_KEY'] = KEY_OPEN_AI

import re
import openai


def safe_text(doc) -> str:
    """Convert dict/Document/str to safe text"""
    if doc is None:
        return ""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return (
                doc.get("text")
                or doc.get("page_content")
                or doc.get("content")
                or doc.get("body")
                or ""
        )
    if hasattr(doc, "page_content"):
        return getattr(doc, "page_content", "") or ""
    if hasattr(doc, "text"):
        return getattr(doc, "text", "") or ""
    return str(doc)  # fallback


class FastOpenAIReranker:
    def __init__(self, api_key: str = None):
        self.client = openai.AsyncOpenAI(api_key=KEY_OPEN_AI)
        self.model = GPT_5_1

        # fix1: local cross-encoder for reranking without an API call
        self._cross_encoder = None
        try:
            from framework.Rankers.CrossEncoderReranker import _get_reranker
            self._cross_encoder = _get_reranker()
        except Exception:
            pass

        # Spanish stopwords: used to filter Spanish query terms (input data, not prose)
        self.stopwords = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'al', 'a', 'en', 'con', 'por', 'para',
            'es', 'son', 'está', 'están', 'fue', 'fueron',
            'y', 'e', 'o', 'u', 'ni', 'que', 'cual', 'cuales',
            'qué', 'cómo', 'cuál', 'cuáles', 'ser', 'estar'
        }

        model_rerank = "cross-encoder (local)" if self._cross_encoder else self.model
        print(f"[FAST RERANK] Ready with {model_rerank}")

    # ════════════════════════════════════════════════════════════════════════
    # MAIN METHOD — rerank_documents
    # FIX 1: effective_top_k forced to a real subset (not all docs)
    # FIX 2: prompt changed from "select" to "order by relevance"
    # ════════════════════════════════════════════════════════════════════════

    async def rerank_documents(
                self,
                query: str,
                documents: list,
                top_k: int = 20,
                filter_noise: bool = True
        ) -> list:
            try:
                print(f"\n{'═' * 80}")
                print(f"[RERANK] 🔄 Reranking with {self.model}")
                print(f"[RERANK] 📊 Input docs: {len(documents)} → Target top_k: {top_k}")
                print(f"[RERANK] 🧹 Noise filter: {'Enabled' if filter_noise else 'Disabled'}")
                print(f"{'═' * 80}\n")

                # ── Basic validations ───────────────────────────────────────────
                if not documents:
                    print(f"[RERANK] ⚠️ No documents to rerank")
                    return []

                # fix1: use the local cross-encoder if available (~50ms vs 1-3s LLM)
                if self._cross_encoder is not None:
                    try:
                        import asyncio
                        from framework.Rankers.CrossEncoderReranker import rerank
                        result = await asyncio.wait_for(
                            asyncio.to_thread(rerank, query, documents, top_k),
                            timeout=8.0,
                        )
                        print(f"[RERANK] ✅ Cross-encoder: {len(documents)}→{len(result)} docs (~50ms)")
                        return result
                    except Exception as e:
                        print(f"[RERANK] ⚠️ Cross-encoder failed ({e}), using LLM")

                if len(documents) <= 3:
                    print(f"[RERANK] ⚠️ Too few docs ({len(documents)}), no rerank")
                    return documents

                # ── effective_top_k is always a real subset ─────────────────────
                max_returnable = max(5, int(len(documents) * 0.6))
                effective_top_k = min(top_k, max_returnable)

                print(f"[RERANK] 🎯 effective_top_k: {effective_top_k} (out of {len(documents)} docs)")

                # ── Limit to 50 docs for GPT ────────────────────────────────────
                docs_for_gpt = documents[:30] if len(documents) > 30 else documents

                if len(documents) > 30:
                    print(f"[RERANK] ⚠️ Limiting from {len(documents)} to 30 docs for GPT")

                # ── Enriched snippets ────────────────────────────────────────────
                enriched_snippets = []
                for i, doc in enumerate(docs_for_gpt):
                    try:
                        text = self._get_text_from_doc(doc)
                        snippet = self._extract_best_snippet(text, query, 100)
                        enriched_snippets.append(f"{i}. {snippet}")
                    except Exception:
                        enriched_snippets.append(f"{i}. [error extracting text]")

                # ── Prompt with an explicit ORDERING instruction ────────────────
                prompt = f"""User query: "{query}"

    Order these {len(docs_for_gpt)} documents from MOST to LEAST relevant for answering the query.
    Return ONLY the {effective_top_k} most relevant ones, ordered (most relevant first).

    RANKING CRITERIA:
    1. Direct relevance to the query (highest weight)
    2. Coverage of multiple aspects of the query
    3. Specificity over generality
    4. Topical diversity (avoid repeating content)

    DOCUMENTS:
    {chr(10).join(enriched_snippets)}

    REQUIRED: respond ONLY with {effective_top_k} integers separated by commas.
    The FIRST index = MOST relevant document.
    Do NOT include text, explanations, or the {len(docs_for_gpt)} indices in natural order.
    You MUST respond with numbers. Example: 3,7,0,15,2"""

                print(f"[RERANK] 📤 Prompt: {len(prompt)} chars, {len(enriched_snippets)} docs")

                # ── Call to GPT-5.1 ─────────────────────────────────────────────
                start = time.time()

                response = openai.chat.completions.create(
                    model=GPT_5_4,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an information-retrieval expert. "
                                "Your task is to ORDER documents by relevance to a query. "
                                "Respond ONLY with numeric indices separated by commas, "
                                "from most to least relevant. NEVER return all indices in natural order."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=1000,
                    timeout=40,
                    reasoning_effort="none"
                )

                elapsed = time.time() - start
                result = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

                print(f"[RERANK] ✅ GPT responded in {elapsed:.2f}s")
                print(f"[RERANK] 📥 Raw response: {result[:120]}...")

                # ── Detect a useless response BEFORE parsing ────────────────────
                numbers_in_response = re.findall(r'\d+', result)
                useless_response = not result or result.strip("., ") == "" or len(numbers_in_response) == 0

                if useless_response:
                    print(f"[RERANK] ⚠️ GPT returned a useless response ('{result[:30]}') — activating keyword fallback")
                    indices = self._keyword_fallback_rank(query, docs_for_gpt, effective_top_k)
                else:
                    # ── Parse indices ─────────────────────────────────────────────
                    indices = self._parse_indices_robust(result, max_index=len(docs_for_gpt) - 1)

                    # ── Detect no-op: 0,1,2,...,N in natural order ──────────────
                    is_noop = (
                            len(indices) >= len(docs_for_gpt) and
                            indices == list(range(len(indices)))
                    )
                    if is_noop:
                        print(f"[RERANK] ⚠️ GPT returned natural order (no-op) — activating keyword fallback")
                        indices = self._keyword_fallback_rank(query, docs_for_gpt, effective_top_k)

                    if not indices:
                        print(f"[RERANK] ⚠️ Error parsing indices, using sequential fallback")
                        indices = list(range(min(effective_top_k, len(docs_for_gpt))))

                print(f"[RERANK] 📊 Parsed indices: {len(indices)} → {indices[:10]}...")

                # ── Fill in if GPT returned fewer than effective_top_k ──────────
                if len(indices) < effective_top_k:
                    print(f"[RERANK] ⚠️ Only {len(indices)} indices, filling up to {effective_top_k}")
                    seen = set(indices)
                    for i in range(len(docs_for_gpt)):
                        if i not in seen:
                            indices.append(i)
                        if len(indices) >= effective_top_k:
                            break

                # Limit to effective_top_k
                indices = indices[:effective_top_k]

                # ── Map to documents ─────────────────────────────────────────────
                reranked = [docs_for_gpt[i] for i in indices if 0 <= i < len(docs_for_gpt)]

                print(f"[RERANK] ✅ Reranking complete: {len(reranked)} docs")

                # ── Post-rerank noise filtering (GENEROUS) ───────────────────────
                # Only filters if there are many docs AND the filter is not aggressive.
                # If the filter removes more than 40%, it is discarded entirely
                # and we trust the reranker's order. The extractor (GPT-5.1)
                # already decides which chunks are useful — no need for a double filter.
                if filter_noise and len(reranked) >= 10:
                    reranked_filtered = self._filter_noise_after_rerank(
                        reranked, query, min_keyword_matches=1
                    )
                    ratio = len(reranked_filtered) / len(reranked) if reranked else 0

                    if ratio >= 0.40:
                        print(
                            f"[RERANK] ✅ Filter applied: {len(reranked)} → {len(reranked_filtered)} docs ({ratio * 100:.1f}%)")
                        reranked = reranked_filtered
                    else:
                        print(
                            f"[RERANK] ⚠️ Filter too aggressive ({ratio * 100:.1f}%), keeping full reranked set ({len(reranked)} docs)")
                        # Do NOT filter — trust the reranker + extractor
                else:
                    print(f"[RERANK] ⏭️ Filter skipped (few docs: {len(reranked)}), keeping full reranked set")

                print(f"{'═' * 80}\n")
                return reranked

            except Exception as e:
                print(f"[RERANK] ❌ ERROR: {type(e).__name__}: {str(e)[:300]}")
                import traceback
                traceback.print_exc()
                fallback_limit = min(len(documents), max(top_k * 2, 30))
                print(f"[RERANK] ⚠️ Fallback: returning {fallback_limit} docs without rerank\n")
                return documents[:fallback_limit]

    # ════════════════════════════════════════════════════════════════════════
    # FALLBACK: keyword-based ranking when GPT does a no-op
    # ════════════════════════════════════════════════════════════════════════

    def _keyword_fallback_rank(self, query: str, documents: list, top_k: int) -> list:
        """
        Emergency ranking based on the frequency of query keywords.
        Used when GPT returns natural order.
        """
        keywords, entities = self._extract_keywords_from_query(query)
        scored = []
        for i, doc in enumerate(documents):
            text = self._get_text_from_doc(doc).lower()
            score = (
                sum(1 for kw in keywords if kw in text) * 1.0 +
                sum(2 for ent in entities if ent.lower() in text)
            )
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored[:top_k]]

    # ════════════════════════════════════════════════════════════════════════
    # HELPER METHODS (unchanged from the original)
    # ════════════════════════════════════════════════════════════════════════

    def _extract_keywords_from_query(self, query: str):
        # Spanish stopwords: input-matching data, kept in Spanish on purpose
        STOPWORDS_ES = {
            "que", "qué", "para", "por", "con", "una", "uno", "los", "las",
            "del", "sus", "ante", "frente", "debe", "tener", "hay", "ser",
            "como", "este", "esta", "ese", "esa", "son", "han", "los", "las",
            "más", "muy", "también", "sino", "pero", "desde", "hacia", "sobre",
            "entre", "cuando", "donde", "cuál", "cual", "quién", "quien",
            "cómo", "cuánto", "tanto", "cada", "todo", "toda", "todos",
            "estas", "estos", "aquel", "aquella", "aquellos", "aquellas",
            "dicho", "dicha", "tener", "tiene", "tiene", "tiene", "tiene",
            "puede", "podría", "deben", "debería", "debe", "hacer", "haber",
            "sido", "sido", "está", "están", "era", "eran", "fue", "fueron",
            "así", "aunque", "mientras", "durante", "mediante", "según",
            "pues", "porque", "sino", "sólo", "solo", "vez", "veces",
        }

        stopwords_combined = self.stopwords | STOPWORDS_ES

        query_lower = query.lower()
        all_words = re.findall(r'\b\w{3,}\b', query_lower)
        keywords = set(w for w in all_words if w not in stopwords_combined)
        entities = set(re.findall(r'\b[A-Z]\w{2,}\b|\b\d+\b', query))
        return keywords, entities


    def _filter_noise_after_rerank(
            self,
            documents: list,
            query: str,
            min_keyword_matches: int = 1
    ) -> list:
        print(f"\n[FILTER] 🧹 Filtering post-rerank noise...")
        print(f"[FILTER] 📊 Docs before: {len(documents)}")

        if not documents:
            return []

        keywords, entities = self._extract_keywords_from_query(query)
        print(f"[FILTER] 🔍 Keywords: {', '.join(list(keywords)[:8])}")
        if entities:
            print(f"[FILTER] 🎯 Entities: {', '.join(list(entities)[:8])}")

        filtered = []
        for i, doc in enumerate(documents):
            text = self._get_text_from_doc(doc)
            text_lower = text.lower()
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            entity_matches = sum(1 for ent in entities if ent in text)
            keep = (
                keyword_matches >= min_keyword_matches or
                entity_matches > 0 or
                (len(text) > 400 and keyword_matches > 0) or
                (not entities and keyword_matches > 0)
            )
            if keep:
                filtered.append(doc)
                reason = f"{keyword_matches}kw" + (f" + {entity_matches}ent" if entity_matches else "")
                preview = text[:80].replace('\n', ' ')
                print(f"[FILTER]    ✅ [{i}] {reason}: {preview}...")
            else:
                preview = text[:80].replace('\n', ' ')
                print(f"[FILTER]    ❌ [{i}] {keyword_matches}kw {entity_matches}ent: {preview}...")

        print(f"[FILTER] ✅ Docs after: {len(filtered)}")
        print(f"[FILTER] 📊 Reduction: {len(documents) - len(filtered)} docs")
        if documents:
            ratio = len(filtered) / len(documents) * 100
            print(f"[FILTER] 📈 Ratio: {ratio:.1f}%\n")

        return filtered

    def _get_text_from_doc(self, doc) -> str:
        try:
            if hasattr(doc, 'text'):
                return doc.text
            if isinstance(doc, dict):
                return doc.get('text', doc.get('content', ''))
            if isinstance(doc, str):
                return doc
            return str(doc)
        except:
            return ""

    def _parse_indices_robust(self, text: str, max_index: int) -> list:
        try:
            numbers = re.findall(r'\d+', text)
            indices = []
            seen = set()
            for num_str in numbers:
                try:
                    idx = int(num_str)
                    if 0 <= idx <= max_index and idx not in seen:
                        indices.append(idx)
                        seen.add(idx)
                except:
                    continue
            return indices
        except Exception as e:
            print(f"[RERANK] ⚠️ Error parsing indices: {e}")
            return []

    def _extract_best_snippet(self, text: str, query: str, max_length: int = 150) -> str:
        try:
            if not text:
                return "[empty]"
            if len(text) <= max_length:
                return text
            query_lower = query.lower()
            text_lower = text.lower()
            best_pos = 0
            query_words = query_lower.split()[:5]
            for word in query_words:
                pos = text_lower.find(word)
                if pos != -1:
                    best_pos = max(0, pos - 50)
                    break
            snippet = text[best_pos:best_pos + max_length]
            if best_pos > 0:
                snippet = "..." + snippet
            if best_pos + max_length < len(text):
                snippet = snippet + "..."
            return snippet.strip()
        except Exception:
            return text[:max_length] + "..." if len(text) > max_length else text

    # ════════════════════════════════════════════════════════════════════════
    # SYNC METHODS (unchanged — kept for compatibility)
    # ════════════════════════════════════════════════════════════════════════

    def _coarse_pass_sync(self, query: str, documents: list, target: int) -> list:
        """SYNCHRONOUS Coarse Pass — unchanged from the original"""
        try:
            print(f"\n{'=' * 80}")
            print(f"[RERANK SYNC] 🔄 SYNCHRONOUS Coarse Pass")
            print(f"[RERANK SYNC] 📊 Docs: {len(documents)} → Target: {target}")
            print(f"{'=' * 80}\n")

            if len(documents) <= target:
                print(f"[RERANK SYNC] ⚠️ Few docs, no Coarse Pass")
                return documents

            max_docs = min(len(documents), 60)
            docs_to_process = documents[:max_docs]

            summaries = []
            for i, doc in enumerate(docs_to_process):
                try:
                    text = self._get_text_from_doc(doc)
                    summary = text[:100] if text else "[empty]"
                    summaries.append(f"{i}. {summary}")
                except Exception:
                    summaries.append(f"{i}. [error]")

            if len(summaries) < 5:
                return documents[:target]

            effective_target = min(target, len(summaries) - 2)

            prompt = f"""Query: "{query}"

From these {len(summaries)} documents, select the {effective_target} MOST relevant.

{chr(10).join(summaries)}

Respond ONLY with numbers separated by commas (no spaces, no extra text).
Example: 0,2,5,8,12,15,20,25"""

            start = time.time()
            response = openai.chat.completions.create(
                model=GPT_5_4,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1000,
                reasoning_effort="none",
                timeout=30
            )
            elapsed = time.time() - start
            print(f"[RERANK SYNC] ✅ Response in {elapsed:.2f}s")

            response_text = response.choices[0].message.content.strip()
            indices = self._parse_indices_robust(response_text, max_index=len(docs_to_process) - 1)

            if not indices:
                return documents[:target]

            selected = [docs_to_process[i] for i in indices if 0 <= i < len(docs_to_process)]

            if len(selected) < target // 2:
                selected_set = set(id(doc) for doc in selected)
                remaining = [doc for doc in docs_to_process if id(doc) not in selected_set]
                needed = target - len(selected)
                selected = selected + remaining[:needed]

            print(f"[RERANK SYNC] ✅ Coarse Pass OK: {len(selected)} docs")
            return selected

        except Exception as e:
            print(f"\n[RERANK SYNC] ❌ ERROR: {type(e).__name__}: {str(e)[:500]}")
            return documents[:target]

    def _fine_pass_sync(self, query: str, documents: list, critical_indices: set, top_k: int) -> list:
        """SYNCHRONOUS Fine Pass — unchanged from the original"""
        try:
            print(f"\n{'=' * 80}")
            print(f"[RERANK SYNC] 🔍 SYNCHRONOUS Fine Pass")
            print(f"[RERANK SYNC] 📊 Docs: {len(documents)} → Top: {top_k}")
            print(f"{'=' * 80}\n")

            if len(documents) > 30:
                critical_docs = [documents[i] for i in critical_indices if i < len(documents)]
                non_critical = [doc for i, doc in enumerate(documents) if i not in critical_indices]
                documents = critical_docs + non_critical
                documents = documents[:30]

            enriched = []
            for i, doc in enumerate(documents):
                try:
                    text = self._get_text_from_doc(doc)
                    snippet = self._extract_best_snippet(text, query, 150)
                    enriched.append(f"{i}. {snippet}")
                except Exception:
                    enriched.append(f"{i}. [error]")

            prompt = f"""Query: "{query}"

Rank these {len(enriched)} documents by relevance.

{chr(10).join(enriched)}

Return ONLY the {top_k} most relevant indices, separated by commas:"""

            start = time.time()
            response = openai.chat.completions.create(
                model=GPT_5_4,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000,
                timeout=30,
                reasoning_effort="none"
            )
            elapsed = time.time() - start
            print(f"[RERANK SYNC] ✅ Response in {elapsed:.2f}s")

            result = response.choices[0].message.content.strip()
            indices = self._parse_indices_robust(result, max_index=len(documents) - 1)

            if not indices:
                critical_docs = [documents[i] for i in critical_indices if i < len(documents)]
                remaining = [doc for i, doc in enumerate(documents) if i not in critical_indices]
                return (critical_docs + remaining)[:top_k]

            indices_set = set(indices)
            critical_missing = [i for i in critical_indices if i < len(documents) and i not in indices_set]
            if critical_missing:
                indices = list(critical_missing) + [i for i in indices if i not in critical_indices]

            indices = indices[:top_k]

            if len(indices) < top_k:
                all_indices = set(range(len(documents)))
                missing_indices = sorted(all_indices - set(indices))
                needed = top_k - len(indices)
                indices.extend(missing_indices[:needed])

            merged = [documents[i] for i in indices if 0 <= i < len(documents)]
            print(f"[RERANK SYNC] ✅ Fine Pass OK: {len(merged)} final docs")
            return merged

        except Exception as e:
            print(f"[RERANK SYNC] ❌ ERROR: {type(e).__name__}: {str(e)[:300]}")
            critical_docs = [documents[i] for i in critical_indices if i < len(documents)]
            remaining = [doc for i, doc in enumerate(documents) if i not in critical_indices]
            return (critical_docs + remaining)[:top_k]

    def _identify_critical_content(self, query: str, documents: list) -> set:
        """Generic identification of critical content — unchanged"""
        print(f"\n[CRITICAL] 🔍 Identifying critical content for {len(documents)} docs")
        critical = set()
        if not documents:
            return critical

        snippets = []
        for i, doc in enumerate(documents):
            try:
                text = self._get_text_from_doc(doc)
                text = ' '.join(text.split())
                snippet = text[:250] if len(text) > 250 else text
                snippets.append({'index': i, 'snippet': snippet})
            except Exception:
                snippets.append({'index': i, 'snippet': '[error]'})

        if len(snippets) > 40:
            snippets = snippets[:40]

        doc_list = [f"{s['index']}. {s['snippet']}" for s in snippets]

        prompt = f"""Analyze which documents are CRITICAL to fully answer this question:

QUESTION: "{query}"

DOCUMENTS:
{chr(10).join(doc_list)}

Respond in JSON:
{{
  "aspectos_identificados": ["aspect1", "aspect2"],
  "documentos_criticos": [0, 5, 12],
  "razonamiento": "brief explanation"
}}"""

        try:
            start_time = time.time()
            response = openai.chat.completions.create(
                model=GPT_5_4,
                messages=[
                    {"role": "system", "content": "You are an information-analysis expert. Respond only in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000,
                reasoning_effort="none"
            )
            elapsed = time.time() - start_time
            result = json.loads(response.choices[0].message.content)
            print(f"[CRITICAL] ✅ GPT responded in {elapsed:.2f}s")

            docs_criticos = result.get('documentos_criticos', [])
            valid_indices = [idx for idx in docs_criticos if isinstance(idx, int) and 0 <= idx < len(documents)]
            critical.update(valid_indices)

        except Exception as e:
            print(f"[CRITICAL] ❌ Error: {e} — fallback to first 5 docs")
            critical.update(range(min(5, len(documents))))

        min_docs = min(3, len(documents))
        if len(critical) < min_docs:
            additional = [i for i in range(len(documents)) if i not in critical]
            critical.update(additional[:min_docs - len(critical)])

        print(f"[CRITICAL] ✅ Total critical: {len(critical)}/{len(documents)} → {sorted(list(critical))}")
        return critical

    def _merge_results(self, gpt_response: str, documents: list, critical_indices: set, top_k: int) -> list:
        try:
            indices = self._parse_indices_robust(gpt_response, max_index=len(documents) - 1)
            if not indices:
                critical_docs = [documents[i] for i in critical_indices if i < len(documents)]
                remaining = [doc for i, doc in enumerate(documents) if i not in critical_indices]
                return (critical_docs + remaining)[:top_k]
            indices_set = set(indices)
            critical_missing = [i for i in critical_indices if i < len(documents) and i not in indices_set]
            if critical_missing:
                indices = list(critical_indices) + [i for i in indices if i not in critical_indices]
            return [documents[i] for i in indices[:top_k] if 0 <= i < len(documents)]
        except Exception as e:
            print(f"[RERANK] ⚠️ Error during merge: {e}")
            return documents[:top_k]

    def _parse_indices(self, text: str) -> list:
        numbers = re.findall(r'\d+', text)
        return [int(n) for n in numbers]

    def _create_fast_summary(self, documents: list) -> str:
        summaries = []
        for i, doc in enumerate(documents):
            doc_text = self._get_text_from_doc(doc)
            filename = self._extract_filename_fast(doc_text)
            content = self._extract_key_content(doc_text, max_chars=100)
            summaries.append(f"{i}: {filename} - {content}")
            if len(summaries) >= 15:
                break
        return '\n'.join(summaries)

    def _extract_filename_fast(self, text: str) -> str:
        lines = text.split('\n')[:3]
        for line in lines:
            if 'ARCHIVO:' in line or '.pdf' in line:
                parts = line.split()
                for part in parts:
                    if '.pdf' in part or '.doc' in part:
                        return part
                return line.split(':')[-1].strip()[:30]
        return "doc"

    def _extract_key_content(self, text: str, max_chars: int = 100) -> str:
        for line in text.split('\n'):
            line = line.strip()
            if (len(line) > 20 and
                    not line.startswith('===') and
                    not line.startswith('---') and
                    ':' not in line[:10]):
                return line[:max_chars] + "..." if len(line) > max_chars else line
        return "document"

    def _fast_reorder(self, result: str, documents: list, top_k: int) -> list:
        try:
            numbers = re.findall(r'\d+', result)
            ranked_ids = [int(n) for n in numbers if int(n) < len(documents)]
            reranked, used = [], set()
            for doc_id in ranked_ids:
                if doc_id not in used and len(reranked) < top_k:
                    reranked.append(documents[doc_id])
                    used.add(doc_id)
            for i, doc in enumerate(documents):
                if i not in used and len(reranked) < top_k:
                    reranked.append(doc)
            return reranked
        except Exception:
            return documents[:top_k]
