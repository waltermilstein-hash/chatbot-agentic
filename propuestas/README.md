# Propuestas de trabajo — Universidad de San Luis · IA

Directorio de **propuestas de trabajo final / tesina** sobre la base de este
proyecto (**Chatbot RAG + 🐎 Pegasus** + infra dockerizada `unsl-tesis`).
Pensadas como puntos de partida para un **ingeniero/a próximo/a a recibirse**:
cada una es autocontenida, acotable a un cuatrimestre y reutiliza la infra y el
código ya existentes.

> **Encuadre.** Todo el trabajo es **académico y defensivo**, ejecutado sobre la
> **propia infra en loopback (`127.0.0.7` / `127.0.0.1`)**. Las propuestas de
> seguridad (prompt injection, red-teaming) son **pruebas autorizadas sobre el
> sistema propio**, no contra terceros. Ver la nota ética en cada archivo.

## Índice

| # | Propuesta | Eje | Dificultad |
|---|-----------|-----|------------|
| [01](01-resiliencia-ablaciones-caos.md) | Resiliencia de la infra + chatbot por **ablaciones** e **ingeniería de caos** | Confiabilidad | Media-Alta |
| [02](02-evaluacion-rag-calidad.md) | Evaluación sistemática de **calidad del RAG** (retrieval + faithfulness) | Calidad / ML | Media |
| [03](03-agente-prompt-injection.md) | **Agente de prompt injection** sobre los endpoints de IA | Seguridad (ofensiva/defensiva) | Alta |
| [04](04-red-team-guardrails-compliance.md) | **Red-team de guardrails** y evaluación del pipeline de *compliance* (AI-Act / VERA / BVG) | Seguridad / Gobernanza | Alta |
| [05](05-observabilidad-slos.md) | **Observabilidad y SLOs** end-to-end (OTel / Prometheus / Grafana / Tempo) | SRE / Operación | Media |
| [06](06-benchmark-modelos-costo-latencia.md) | **Benchmark de modelos**: costo, latencia y calidad por configuración | ML Ops / Economía | Media |

## Cómo está armada cada propuesta

Cada archivo sigue la misma estructura para que sea directamente defendible:

1. Contexto y motivación
2. Objetivo general y objetivos específicos
3. Alcance (incluye / excluye)
4. Metodología
5. Stack y herramientas (de la infra existente)
6. Métricas / KPIs
7. Entregables
8. Cronograma estimado (12–16 semanas)
9. Criterios de éxito
10. Riesgos y consideraciones éticas

## Recursos del proyecto reutilizables

- **App única** (`main_api.py`): endpoints `/virtualbot/...` (RAG) y `/api/v1/agent/web...` (Pegasus).
- **Motor RAG** (`chatbot/pipeline_v2/`): orquestador, router de intención, búsqueda vectorial + multi-query, filtrado, citas y **compliance** (AI-Act, VERA, BVG, detector de contradicciones, audit trail).
- **Infra `unsl-tesis`** (Docker, `127.0.0.7`): Weaviate, Mongo, Redis, MinIO, **Kafka, OpenSearch, Graylog, Prometheus, Tempo, OTel Collector, Grafana, MLflow, Traefik** y un **pentest toolbox**.
- **UI** (`ui/unsl_ui.py`): banco de pruebas chainlit.
