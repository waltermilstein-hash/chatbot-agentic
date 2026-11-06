# Universidad de San Luis · IA — Chatbot RAG + 🐎 Pegasus

Aplicación **única** (un solo proceso FastAPI) que reúne dos capacidades:

- **Chatbot RAG documental** — responde sobre los documentos cargados, con
  recuperación vectorial, filtrado, citas y una capa de *compliance*.
- **🐎 Pegasus · navegación autónoma** — agente **ReAct sobre LangGraph** que
  busca y lee la web, y sintetiza una respuesta con citas.

> **Uso académico.** Este proyecto es material de cátedra / tesis de la
> **Universidad de San Luis (UNSL)**. Está pensado como una **base simple y
> potente** —chatbot + agente + infra en Docker— para que el alumno la
> **extienda**; no es un producto de producción. Corre en *loopback*, sin
> autenticación, como demo de cátedra.
>
> **Co-directora:** Leticia Cagnina.

---

## Arquitectura

```
Agentica/
├── main_api.py              # App ÚNICA FastAPI: monta chatbot (/virtualbot) + Pegasus (/api/v1)
├── run.sh                   # Arranque del backend (venv + .env)  →  :8080
│
├── app/
│   ├── api/ChatbotService.py   # Router del chatbot RAG + cliente OpenAI
│   └── middleware/             # Middleware HTTP
│
├── chatbot/
│   └── pipeline_v2/            # 🔧 MOTOR RAG (RAGPipeline) — NO es un fallback
│       ├── orchestrator.py        # orquesta el flujo de respuesta
│       ├── intent_router.py       # enruta por intención
│       ├── search/                # vector_search + multi_query
│       ├── filtering/             # chunk_filter (poda de contexto)
│       ├── response/              # citation (respuesta con citas)
│       ├── response_modes.py / context_manager.py / llm_provider.py
│       └── compliance/            # guardas: AI-Act, VERA multidimensional,
│                                   #  BVG, detector de contradicciones, audit_trail
│
├── agente_web/              # 🐎 Pegasus
│   ├── agent.py                # agente ReAct (LangGraph, create_react_agent)
│   ├── tools.py                # herramientas: buscar_web / leer_pagina + SYSTEM
│   ├── api.py                  # router /api/v1/agent/web (+ /stream SSE)
│   └── ui/agent_ui.py          # UI chainlit solo-agente (opcional)
│
├── framework/               # Cierre de dependencias del chatbot
│   ├── ConfigurationManager/   # loader del config.yml (apunta a la infra)
│   ├── Database/MongoDb · Redis · Weaviate   # conectores a la infra
│   ├── Embeddings · Rankers · Searchers · Splitters   # pipeline documental
│   ├── Export · ORC(OCR) · Users · chatbotValidators · StringNormalize
│
├── config/ · configyml/ · Constants/        # configuración y constantes
│
├── ui/
│   └── unsl_ui.py           # UI chainlit ÚNICA  →  :8501
│                            #   mensaje normal → RAG · `/web <texto>` → Pegasus
│
└── dockers/docker-platforms/   # INFRA (solo contenedores, NO la app)
    └── docker-compose.yml       # Weaviate · Mongo · Redis · MinIO en 127.0.0.7
```

**La app corre FUERA de Docker (en el host).** Los contenedores son únicamente
infraestructura y se exponen en `127.0.0.7`:

| Servicio  | Dirección          | Uso                                  |
|-----------|--------------------|--------------------------------------|
| Weaviate  | `127.0.0.7:8087`   | base vectorial (chunks + embeddings) |
| MongoDB   | `127.0.0.7:27017`  | conversaciones / tópicos             |
| Redis     | `127.0.0.7:6379`   | caché / estado                       |
| MinIO     | `127.0.0.7:9000-1` | objetos / adjuntos                   |

### Flujo (resumido)

- **Chatbot RAG:** petición → `ChatbotService` → **RAGPipeline** (`pipeline_v2`):
  `intent_router` → `search` (vectorial + multi-query sobre Weaviate) →
  `filtering` → `response` con citas → **compliance** (AI-Act / VERA / BVG /
  contradicciones / *audit trail*) → stream SSE.
- **🐎 Pegasus:** petición → agente **ReAct** (LangGraph) → el LLM decide
  `buscar_web` / `leer_pagina`, lee lo devuelto y sintetiza con citas → stream SSE.
  Memoria multi-turno opcional por `thread_id`.

---

## Endpoints

| Método | Ruta | Qué hace |
|--------|------|----------|
| `POST` | `/virtualbot/chatbot/rag/chatbotservice/stream` | Chat RAG (SSE). Multipart `json_data = {user, empresa, topico, prompt, timestamp}` |
| `POST` | `/virtualbot/chatbot/rag/AutoTrainingBotByUser` | Entrena / ingesta documental |
| `GET`  | `/virtualbot/chatbot/export/download/{file_id}` | Descarga un export |
| `POST` | `/api/v1/agent/web` | 🐎 Pegasus (respuesta completa) |
| `POST` | `/api/v1/agent/web/stream` | 🐎 Pegasus (SSE) |
| `GET`  | `/health` | Salud: `{status, app, components}` |

Documentación interactiva: **http://localhost:8080/docs**

---

## Arranque

### 1) Infra (una vez)

```bash
cd dockers/docker-platforms
docker compose up -d          # Weaviate · Mongo · Redis · MinIO → 127.0.0.7
```

### 2) Configuración

```bash
cp .env.example .env          # completá OPENAI_API_KEY (y JWT_SECRET si aplica)
```

El `.env` ya apunta la infra a `127.0.0.7` y define `UNSL_API_BASE`,
`CHATBOT_CONFIG_YML`, modelos del agente, etc.

### 3) Backend (app única)

```bash
./run.sh                      # → http://127.0.0.1:8080   (PORT=9000 ./run.sh para otro puerto)
```

> **Nota:** si los puertos `8080` de la infra Docker ocupan `0.0.0.0:8080`,
> levantá el backend ligado a loopback:
> `.venv/bin/uvicorn main_api:app --host 127.0.0.1 --port 8080`

### 4) UI chainlit (otra terminal)

```bash
set -a && . ./.env && set +a
.venv/bin/chainlit run ui/unsl_ui.py --host 127.0.0.1 --port 8501
```

- **UI:** http://localhost:8501
  - mensaje normal → **Chatbot RAG**
  - `/web <texto>` → **🐎 Pegasus** (navegación autónoma)
- **API docs:** http://localhost:8080/docs

---

## Requisitos

- Python del `venv` del proyecto (`.venv/`) con `requirements.txt` instalado.
- `OPENAI_API_KEY` **obligatoria** para respuestas reales.
- Infra de la tesis levantada en `127.0.0.7` (paso 1).
- Los datos se cargan **a mano por el endpoint de ingesta** (no hay sync con
  SharePoint).
