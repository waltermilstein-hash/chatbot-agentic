# =============================================================================
# =============================================================================
"""
ContextManager — assembling conversational context with compaction.

Instead of sending the whole history to the model:
  - keeps the last `recent_turns` messages as-is,
  - summarizes the earlier ones into a `conversation_summary` block (preserving facts,
    decisions, entities and mentioned documents),
  - respects a character budget.

Returns a list of messages [{role, content}] ready for the LLM.
"""
import logging
from typing import List, Dict, Optional

from config import Constants

logger = logging.getLogger(__name__)


class ConversationState:
    """Lightweight per-conversation state (in process memory) to resolve
    follow-ups: last mode used (documental/general) and last subject/topic.
    It does not replace chatbot_memory; it is a hint so the thread isn't lost."""
    _store: Dict[str, Dict] = {}
    _MAX = 500

    @classmethod
    def set(cls, conversation_id, **kw):
        if not conversation_id:
            return
        if conversation_id not in cls._store and len(cls._store) >= cls._MAX:
            cls._store.pop(next(iter(cls._store)), None)  # memory bound
        cls._store.setdefault(conversation_id, {}).update(
            {k: v for k, v in kw.items() if v is not None}
        )

    @classmethod
    def get(cls, conversation_id) -> Dict:
        return dict(cls._store.get(conversation_id, {}))

    @classmethod
    def mark_documental(cls, conversation_id, subject=None):
        """Mark the turn as documental and set the LAST documental subject/mode.
        This anchor persists across general turns: follow-ups return to it
        (avoids the cascade effect where a slip to general loses the thread)."""
        cls.set(conversation_id, last_mode="documental",
                last_documental_mode="documental", last_documental_subject=subject)

    @classmethod
    def mark_general(cls, conversation_id):
        """Mark the turn as general WITHOUT touching the documental anchor (subject/mode),
        which is kept to re-anchor the next follow-up."""
        cls.set(conversation_id, last_mode="general")

    @classmethod
    def documental_subject(cls, conversation_id):
        return cls.get(conversation_id).get("last_documental_subject")


class ContextManager:
    def __init__(self, recent_turns: int = 6, char_budget: int = 6000, summary_model: Optional[str] = None):
        self.recent_turns = recent_turns
        self.char_budget = char_budget
        self.summary_model = summary_model or getattr(Constants, "GPT_5_NANO", "gpt-5-nano")

    async def _summarize(self, older: List[Dict]) -> str:
        text = "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in older)[:8000]
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=Constants.KEY_OPEN_AI)
            resp = await client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": (
                        "Summarize the conversation preserving: facts, decisions, entities, "
                        "mentioned documents and open questions. Be brief and factual."
                    )},
                    {"role": "user", "content": text},
                ],
                max_completion_tokens=300,
                reasoning_effort="low",
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ContextManager summarize failed: %s", exc)
            # Fallback: simple truncation.
            return text[:1000]

    async def build(self, history: Optional[List[Dict]], query: str,
                    system_prompt: str) -> List[Dict]:
        history = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
        messages: List[Dict] = [{"role": "system", "content": system_prompt}]

        if len(history) > self.recent_turns:
            older = history[: -self.recent_turns]
            recent = history[-self.recent_turns:]
            summary = await self._summarize(older)
            if summary:
                messages.append({"role": "system", "content": f"SUMMARY OF THE PREVIOUS CONVERSATION:\n{summary}"})
        else:
            recent = history

        # Respect the character budget over the recent ones (newest to oldest).
        budget = self.char_budget
        kept = []
        for m in reversed(recent):
            c = str(m.get("content", ""))[:2000]
            if budget - len(c) < 0:
                break
            kept.append({"role": m["role"], "content": c})
            budget -= len(c)
        messages.extend(reversed(kept))
        messages.append({"role": "user", "content": query})
        return messages
