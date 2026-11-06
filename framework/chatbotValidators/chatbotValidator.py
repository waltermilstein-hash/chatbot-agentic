"""
Content validator via OpenAI's moderation API (1.x client).
It is a best-effort GUARDRAIL: if moderation cannot run (missing key, network
error, 401...), it must NOT block the query → it returns es_apropiado=None and
the caller treats None as "could not validate" (fail-open), never as
inappropriate.
"""
import logging
import os

from openai import OpenAI

logger = logging.getLogger("chatbotValidator")


def _resolve_key(explicit: str | None = None) -> str:
    # Accepts the project's convention (KEY_OPEN_AI) and OpenAI's standard ones.
    return (explicit or os.getenv("KEY_OPEN_AI") or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_APIKEY") or "")


class ChatbotValidator:
    def __init__(self, api_key: str | None = None):
        self._key = _resolve_key(api_key)
        self._client = OpenAI(api_key=self._key) if self._key else None

    def is_appropriate_content(self, text: str):
        """Return (is_appropriate, categories).

        - (True, scores)  → appropriate content.
        - (False, scores) → content flagged by moderation.
        - (None, {})      → could not validate (no key / error) → fail-open in the caller.
        """
        if not self._client:
            logger.warning("Moderation disabled: no OPENAI API key; skipping the check.")
            return None, {}
        try:
            response = self._client.moderations.create(
                model="omni-moderation-latest", input=text
            )
            result = response.results[0]
            return (not result.flagged), result.category_scores
        except Exception as e:  # noqa: BLE001 - best-effort guardrail
            logger.warning("Could not validate the content (skipping the check): %s", e)
            return None, {}
