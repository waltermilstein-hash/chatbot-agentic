# =============================================================================
# =============================================================================
"""
System configuration constants.

API keys and environment-specific endpoints are read from environment variables
(.env). Do NOT hardcode secrets here: add them to .env and read them with
os.getenv. Model aliases and per-stage routing do live here as constants.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ─── API keys (from .env) ─────────────────────────────────────────────────────
KEY_OPEN_AI = os.getenv("KEY_OPEN_AI") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or ""
DEEP_SEEK   = os.getenv("DEEP_SEEK", "")
ALIBABA     = os.getenv("ALIBABA", "")
HUGGINGFACE = os.getenv("HUGGINGFACE", "")
AZURE_KEY   = os.getenv("AZURE_KEY", "")
URL_AZURE   = os.getenv("URL_AZURE", "")

# ─── Kimi / Moonshot (from .env) ──────────────────────────────────────────────
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_API_URL = os.getenv("KIMI_API_URL", "https://api.moonshot.ai/v1/chat/completions")

# ─── Google Gemini (from .env) ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── OpenAI ───────────────────────────────────────────────────────────────
GPT_5           = "gpt-5"
GPT_5_1         = "gpt-5.1"
GPT_5_NANO      = "gpt-5-nano"
GPT_5_4         = "gpt-5.4"
GPT_5_4_MINI    = "gpt-5.4-mini"
GPT_5_6         = "gpt-5.6"
# ─── Google ───────────────────────────────────────────────────────────────
GEMINI_3_1_PRO  = "gemini-3.1-pro-preview"
GEMINI_2_5_PRO  = "gemini-2.5-pro"
# ─── Anthropic ────────────────────────────────────────────────────────────
CLAUDE_SONNET   = "claude-sonnet-4-6"
CLAUDE_OPUS     = "claude-opus-4-6"
CLAUDE_HAIKU    = "claude-haiku-4-5-20251001"

# ─── Per-stage assignment ──────────────────────────────────────────────────
MODEL_COREFERENCE   = GPT_5_1
MODEL_ROUTER        = GPT_5_1
MODEL_DECOMPOSER    = GPT_5_4_MINI
MODEL_REFORMULATION = GPT_5_4_MINI
MODEL_SYNTHESIS     = GEMINI_3_1_PRO
