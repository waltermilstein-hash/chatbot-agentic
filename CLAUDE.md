# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Academic project (UNSL thesis, "Universidad de San Luis · IA"). A **single FastAPI process** that serves two capabilities, mounted as two routers in `main_api.py`:

- **Chatbot RAG** (`/virtualbot`, from `app/api/ChatbotService.py`) — document RAG over Weaviate with retrieval, filtering, citations, and a compliance layer.
- **🐎 Pegasus** (`/api/v1`, from `agente_web/`) — a ReAct agent on LangGraph that searches/reads the web and synthesizes with citations.

The code and all docs (README.md, INSTALACION.md, inline comments) are in **Spanish**. It's a cátedra/thesis demo: runs on loopback, no authentication, not production.

## Running

The **app runs on the host, OUTSIDE Docker**. Docker provides only infrastructure, bound to the unusual loopback IP `127.0.0.7`:

| Service  | Address           |
|----------|-------------------|
| Weaviate | `127.0.0.7:8087`  |
| MongoDB  | `127.0.0.7:27017` |
| Redis    | `127.0.0.7:6379`  |
| MinIO    | `127.0.0.7:9000-1`|

```bash
# 1) Infra (once)
cd dockers/docker-platforms && docker compose up -d

# 2) Config — .env is required and gitignored (holds secrets)
cp .env.example .env   # then fill OPENAI_API_KEY and fix CHATBOT_CONFIG_YML

# 3) Backend (port 8080; PORT=9000 ./run.sh for another)
./run.sh

# 4) UI (separate process, separate terminal)
set -a && . ./.env && set +a
.venv/bin/chainlit run ui/unsl_ui.py --host 127.0.0.1 --port 8501
```

- Backend API docs: `http://localhost:8080/docs`; health at `GET /health`.
- UI at `http://localhost:8501`: a normal message → RAG; `ui/unsl_ui.py` also regex-routes web-intent queries (and explicit `/web <text>`) to Pegasus.
- There is **no test suite, linter config, or build system** (no pytest/Makefile/pyproject). Python **3.11** in `.venv/`; `run.sh` calls `.venv/bin/uvicorn` directly. Install from `requirements.txt` (or `requirements.lock.txt` for pinned versions).

## Environment gotchas

- `CHATBOT_CONFIG_YML` must be the **absolute path** to this repo's `config/config.yml`. `.env.example` and the docs still reference an old `/home/athenas/DESARROLLO/Agentica/...` path — the actual repo dir is `chatbot-agentic`; adjust it.
- The OpenAI key must be set in **both** `OPENAI_API_KEY` and `OPENAI_APIKEY` (different code paths read different names; `config.Constants.KEY_OPEN_AI` falls back across `KEY_OPEN_AI → OPENAI_API_KEY → OPENAI_APIKEY`).
- `config/config.yml` is sanitized (no secrets, points infra at `127.0.0.7`); the real OpenAI key comes from the environment, not the yml.

## Architecture — the non-obvious parts

**Two RAG pipeline classes, layered.** `ChatbotService` imports the legacy `chatbot/RAGPipeline.py` (`RAGPipeline`), but the stream endpoint actually instantiates **`RAGPipelineV2`** (`chatbot/pipeline_v2/`, aka the orchestrator). V2 is the real entry point: it **wraps the legacy `RAGPipeline` and adds the compliance layer**. When tracing the chat flow, start at `chatbot/pipeline_v2/orchestrator.py`, not `RAGPipeline.py`.

**Chat flow:** `ChatbotService.bot_services_stream` → `RAGPipelineV2.stream()` → `intent_router` → `search/` (vector + multi-query on Weaviate) → `filtering/` (chunk pruning) → `response/` (citations) → `compliance/` guards → SSE.

**Compliance guards** live in `chatbot/pipeline_v2/compliance/` (`pipeline_guard.py` orchestrates: AI-Act, VERA multidimensional, BVG, contradiction detector, audit trail). Per `CHATBOT_CLEANUP_REPORT.md`, many optional components are imported inside `try/except` and **degrade gracefully** when absent — a missing import is a feature, not a bug.

**Pegasus** (`agente_web/`): `agent.py` builds a LangGraph `create_react_agent` with an in-memory checkpointer (multi-turn memory keyed by `thread_id`); `tools.py` defines `buscar_web`/`leer_pagina` + the SYSTEM prompt; `api.py` is the router with a `/stream` SSE variant.

**Two separate `Constants` modules** — don't confuse them:
- `config/Constants.py` — API keys from env **and model aliases routed per stage** (`MODEL_ROUTER`, `MODEL_DECOMPOSER`, `MODEL_SYNTHESIS`, etc.). Change model assignments here.
- `Constants/Constants.py` — hardcoded Spanish **prompt strings** (personas, disclaimers, `HTML_FORMAT`).

**`framework/`** holds the document/infra plumbing shared by both endpoints: `Weaviate/`, `Embeddings/`, `Rankers/` (cross-encoder reranker degrades to score order if deps missing), `Searchers/`, `Splitters/`, `ORC/` (**PaddleOCR** — Azure DI was removed), `Database/MongoDb/`, `Redis/`, `ConfigurationManager/` (loads the yml), `Export/` (regex-based intent detection → docx/pdf/pptx/xlsx/md exports).

**Ray** is initialized locally at startup (`main_api.py`), used only by the parallel training/ingestion path; init failure is caught and logged (training won't work, app still starts).

**Ingestion** is manual via `POST /virtualbot/chatbot/rag/AutoTrainingBotByUser` — there is no SharePoint/external sync. Documents are loaded by hand through that endpoint. It branches by type when indexing into Weaviate: **PDFs** go through `train.insert_document` + `ChatbotTrainer.training_chat_bot_buscador` (text via PaddleOCR in `framework/ORC/`), while **Office/text documents** (Excel, Word, plain text) go through `run_document_to_weaviate` in `chatbot/CustomLLMOfficeRetriever.py` (auto-detects type, embeds, indexes). The raw binary is then stored in GridFS (Mongo) in the same transactional scope.
