#!/usr/bin/env bash
# =============================================================================
# Creación de @Raven | ravenrootviel@gmail.com
# Arranca la app ÚNICA (chatbot RAG + agente web) con su venv y su .env.
#   uso:  ./run.sh            (puerto 8080)
#         PORT=9000 ./run.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# cargar variables de entorno (.env)
set -a
[ -f "$HERE/.env" ] && . "$HERE/.env"
set +a
export PYTHONPATH="$HERE"
export CHATBOT_CONFIG_YML="${CHATBOT_CONFIG_YML:-$HERE/config/config.yml}"
export ENABLE_DEV_TOKEN="${ENABLE_DEV_TOKEN:-true}"

exec "$HERE/.venv/bin/uvicorn" main_api:app --host 0.0.0.0 --port "${PORT:-8080}"
