# Manual de instalación — Universidad de San Luis · IA

Guía paso a paso para dejar funcionando la app (**Chatbot RAG + 🐎 Pegasus**) en
un host de desarrollo. La app corre **fuera de Docker** (en el host) y se conecta
a la infra dockerizada en `127.0.0.7`.

> ⚠️ **IMPORTANTE — tenés que crear el archivo `.env`.**
> El repo **no** incluye `.env` (trae secretos). Tenés que copiarlo de
> `.env.example` y completar tus claves **antes de arrancar**. Sin `.env` la app
> no se conecta a la infra ni responde con el LLM. Ver [Paso 3](#3-crear-el-env-obligatorio).

---

## 0) Requisitos previos

| Requisito | Versión / nota |
|-----------|----------------|
| **Python** | 3.11 (el venv del proyecto usa **3.11.18**) |
| **Docker + Docker Compose** | para levantar la infra (Weaviate / Mongo / Redis / MinIO) |
| **Clave OpenAI** | `OPENAI_API_KEY` válida — obligatoria para respuestas reales |
| **Puertos libres (loopback)** | `8080` (API) y `8501` (UI) en `127.0.0.1` |

Comprobar:

```bash
python3 --version        # 3.11.x
docker --version
docker compose version
```

---

## 1) Obtener el proyecto

```bash
cd /home/athenas/DESARROLLO/Agentica
```

(Si lo clonás en otra ruta, anotala: la vas a usar en el `.env`, en
`CHATBOT_CONFIG_YML`.)

---

## 2) Entorno Python (venv + dependencias)

Si el repo **ya trae** `.venv/`, podés usarlo directamente. Para crearlo de cero:

```bash
python3.11 -m venv .venv
source .venv/bin/activate         # bash;  en zsh usar:  . ./.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # o requirements.lock.txt para versiones fijas
```

> No hace falta activar el venv para arrancar: `run.sh` usa `.venv/bin/uvicorn`
> directamente.

---

## 3) Crear el `.env` (OBLIGATORIO)

Este paso **no es opcional**. Copiá la plantilla y completá las claves:

```bash
cp .env.example .env
```

Luego **editá `.env`** y completá como mínimo:

```ini
# Ruta ABSOLUTA al config.yml de este repo (ajustala si clonaste en otra carpeta)
CHATBOT_CONFIG_YML=/home/athenas/DESARROLLO/Agentica/config/config.yml

# Secreto JWT (poné una cadena larga propia)
JWT_SECRET=cambia-esto-por-un-secreto-largo-para-la-tesis
ENABLE_DEV_TOKEN=true

# OpenAI — pegá tu clave en AMBAS variables (el código usa los dos nombres)
OPENAI_API_KEY=sk-...tu-clave...
OPENAI_APIKEY=sk-...tu-clave...

# Infra de la tesis (ya apunta a 127.0.0.7; no tocar salvo que cambies puertos)
MONGO_URI=mongodb://127.0.0.7:27017/
REDIS_HOST=127.0.0.7
REDIS_PORT=6379

# UI chainlit → backend
UNSL_API_BASE=http://127.0.0.1:8080
UNSL_EMPRESA=UNSL
UNSL_TOPICO=LEGAL
UNSL_USER=alumno@tesis.local

# Pegasus (agente web)
AGENT_MODEL=gpt-4o-mini
SEARCH_MODEL=gpt-4o-mini
OPENAI_SEARCH_TOOL=web_search_preview
AGENT_RECURSION_LIMIT=25
```

**Checklist del `.env`:**

- [ ] `OPENAI_API_KEY` **y** `OPENAI_APIKEY` con tu clave real.
- [ ] `CHATBOT_CONFIG_YML` apuntando a la ruta **absoluta** de tu copia del repo.
- [ ] `JWT_SECRET` cambiado (no dejar el de ejemplo).
- [ ] `UNSL_API_BASE` = `http://127.0.0.1:8080` (coincide con el puerto del backend).

> 🔒 `.env` queda fuera del control de versiones (ignorado). Nunca lo subas con
> tus claves.

---

## 4) Levantar la infra (Docker)

Desde la carpeta de la infra de la tesis:

```bash
cd dockers/docker-platforms
docker compose up -d
```

Expone en `127.0.0.7` (proyecto compose `unsl-tesis`):

| Servicio | Puerto | Uso |
|----------|--------|-----|
| Weaviate | `8087` | base vectorial |
| MongoDB  | `27017`| conversaciones / tópicos |
| Redis    | `6379` | caché / estado |
| MinIO    | `9000-9001` | objetos / adjuntos |

Verificar que arrancaron:

```bash
docker ps --filter "name=unsl-tesis"
```

Volvé a la raíz del proyecto:

```bash
cd ../..
```

---

## 5) Arrancar el backend (app única)

```bash
./run.sh                 # → http://127.0.0.1:8080
```

> **Si el puerto 8080 da "address already in use":** suele deberse a otros
> contenedores Docker que ocupan `8080` en loopback. Arrancá ligado a
> `127.0.0.1` en vez de `0.0.0.0`:
>
> ```bash
> set -a && . ./.env && set +a
> export PYTHONPATH="$PWD"
> .venv/bin/uvicorn main_api:app --host 127.0.0.1 --port 8080
> ```

Probar que está arriba:

```bash
curl -s http://127.0.0.1:8080/health
# {"status":"ok","app":"unsl-ia","components":["chatbot","agente-web"]}
```

Docs interactivas: **http://localhost:8080/docs**

---

## 6) Arrancar la UI (otra terminal)

```bash
set -a && . ./.env && set +a
.venv/bin/chainlit run ui/unsl_ui.py --host 127.0.0.1 --port 8501
```

Abrir **http://localhost:8501**:

- Mensaje normal → **Chatbot RAG**
- `/web <texto>` → **🐎 Pegasus** (navegación autónoma)

---

## 7) Verificación rápida

| Qué | Cómo | Esperado |
|-----|------|----------|
| Salud API | `curl http://127.0.0.1:8080/health` | `status: ok` |
| Infra | `docker ps --filter name=unsl-tesis` | contenedores `Up (healthy)` |
| UI | navegador → `http://localhost:8501` | carga el chat |
| LLM | log del backend al arrancar | `✅ OpenAI client inicializado` |

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---------|-------|----------|
| `init_openai_client falló` | falta `OPENAI_API_KEY` | completar la clave en `.env` (ambos nombres) y reiniciar |
| `address already in use` en `:8080` | contenedores Docker ocupan `0.0.0.0:8080` | arrancar con `--host 127.0.0.1` (ver Paso 5) |
| La UI no responde / timeout | `UNSL_API_BASE` no coincide con el backend | debe ser `http://127.0.0.1:8080` |
| No encuentra Mongo/Weaviate | la infra no está levantada | `docker compose up -d` en `dockers/docker-platforms` |
| `. .env: no such file` (zsh) | zsh no busca en el dir actual | usar `. ./.env` |

---

**Uso académico** — base de cátedra/tesis de la UNSL, pensada para extenderse.
Co-directora: **Leticia Cagnina**.
