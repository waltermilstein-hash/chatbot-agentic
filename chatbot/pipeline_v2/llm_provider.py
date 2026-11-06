# =============================================================================
# =============================================================================
"""
LLM provider abstraction for the hybrid chatbot.

Common `LLMProvider` interface with OpenAI and Anthropic implementations (both
already present in the stack). Lets you choose provider/model per agent/config
without coupling the pipeline to a specific SDK.

    provider = get_provider("openai")            # or "anthropic"
    async for delta in provider.stream(messages, model="gpt-5.1"):
        ...

`messages` = [{"role": "system"|"user"|"assistant", "content": str}].
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Optional, Tuple

from config import Constants

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def stream(self, messages: List[Dict], model: Optional[str] = None,
                     temperature: float = 0.7, max_tokens: int = 1200) -> AsyncGenerator[str, None]:
        ...

    @abstractmethod
    async def complete(self, messages: List[Dict], model: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 1200) -> str:
        ...

    async def complete_with_usage(
        self, messages: List[Dict], model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: int = 1200,
    ) -> Tuple[str, Optional[Dict]]:
        """Complete and return (text, usage) where `usage` are the REAL tokens
        reported by the provider ({prompt_tokens, completion_tokens, total_tokens})
        or None if not available. The default implementation has no usage
        (reuses `complete`); subclasses override it with a call that exposes it."""
        text = await self.complete(messages, model, temperature, max_tokens)
        return text, None


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = getattr(Constants, "GPT_5_4_MINI", "gpt-5.4-mini")

    def _client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=Constants.KEY_OPEN_AI)

    async def stream(self, messages, model=None, temperature=0.7, max_tokens=1200):
        client = self._client()
        kwargs = {"model": model or self.default_model, "messages": messages, "stream": True}
        # GPT-5 (reasoning) models reject temperature and use max_completion_tokens.
        mdl = (model or self.default_model)
        if str(mdl).startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        resp = await client.chat.completions.create(**kwargs)
        async for chunk in resp:
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            except Exception:
                continue

    async def complete(self, messages, model=None, temperature=0.7, max_tokens=1200):
        out = []
        async for d in self.stream(messages, model, temperature, max_tokens):
            out.append(d)
        return "".join(out)

    async def complete_with_usage(self, messages, model=None, temperature=0.7, max_tokens=1200):
        # NON-streaming call: the response includes `usage` with the real tokens.
        client = self._client()
        mdl = model or self.default_model
        kwargs = {"model": mdl, "messages": messages}
        if str(mdl).startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        resp = await client.chat.completions.create(**kwargs)
        try:
            text = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            text = ""
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
        return text, usage


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = getattr(Constants, "CLAUDE_SONNET", "claude-sonnet-4-6")

    def _key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY") or getattr(Constants, "KEY_ANTHROPIC", None)

    def available(self) -> bool:
        return bool(self._key())

    def _client(self):
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=self._key())

    @staticmethod
    def _split(messages):
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m.get("role") in ("user", "assistant")]
        return system, turns

    async def stream(self, messages, model=None, temperature=0.7, max_tokens=1200):
        system, turns = self._split(messages)
        mdl = model if (model and str(model).startswith("claude")) else self.default_model
        client = self._client()
        async with client.messages.stream(
            model=mdl, system=system or None, messages=turns,
            max_tokens=max_tokens, temperature=temperature,
        ) as s:
            async for text in s.text_stream:
                if text:
                    yield text

    async def complete(self, messages, model=None, temperature=0.7, max_tokens=1200):
        out = []
        async for d in self.stream(messages, model, temperature, max_tokens):
            out.append(d)
        return "".join(out)

    async def complete_with_usage(self, messages, model=None, temperature=0.7, max_tokens=1200):
        # NON-streaming call: the response includes `usage` with the real tokens.
        system, turns = self._split(messages)
        mdl = model if (model and str(model).startswith("claude")) else self.default_model
        client = self._client()
        resp = await client.messages.create(
            model=mdl, system=system or None, messages=turns,
            max_tokens=max_tokens, temperature=temperature,
        )
        try:
            text = "".join(getattr(b, "text", "") for b in (resp.content or [])
                           if getattr(b, "type", None) == "text")
        except Exception:  # noqa: BLE001
            text = ""
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            it = getattr(u, "input_tokens", None)
            ot = getattr(u, "output_tokens", None)
            usage = {
                "prompt_tokens": it,
                "completion_tokens": ot,
                "total_tokens": (it + ot) if (it is not None and ot is not None) else None,
            }
        return text, usage


_PROVIDERS = {"openai": OpenAIProvider, "anthropic": AnthropicProvider}


def get_provider(name: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """Return the provider by name, or infer it from the model name."""
    if not name and model:
        name = "anthropic" if str(model).startswith("claude") else "openai"
    name = (name or "openai").lower()
    cls = _PROVIDERS.get(name, OpenAIProvider)
    inst = cls()
    # If Anthropic is requested but there is no key, fall back to OpenAI.
    if name == "anthropic" and not AnthropicProvider().available():
        logger.warning("AnthropicProvider without API key; using OpenAI.")
        return OpenAIProvider()
    return inst
