# Propuesta 05 — Observabilidad y SLOs end-to-end

**Eje:** SRE / Operación · **Dificultad:** Media · **Duración:** 12–14 semanas

## 1. Contexto y motivación

La infra ya trae el stack de observabilidad montado —**OTel Collector, Tempo,
Prometheus, Grafana, Graylog/OpenSearch**— pero la app del chatbot/Pegasus **no
está completamente instrumentada** ni tiene **SLOs** definidos. Sin esto, las
demás propuestas (resiliencia 01, benchmark 06) no tienen de dónde leer. Esta
propuesta cierra ese hueco y deja la base lista para operar y para que el alumno
vea en vivo qué hace su sistema.

## 2. Objetivos

**General:** instrumentar el sistema end-to-end (trazas, métricas y logs
correlacionados) y definir **SLIs/SLOs** con dashboards y alertas.

**Específicos:**
1. Instrumentar la app con **OpenTelemetry** (trazas distribuidas que crucen app → Weaviate/Mongo/Redis → OpenAI).
2. Exponer **métricas** de negocio y técnicas (latencia por etapa del pipeline, tokens/costo, tasa de error, uso de caché).
3. Centralizar **logs** correlacionados con `trace_id` (Graylog).
4. Definir **SLIs/SLOs** (disponibilidad, latencia, calidad) y **error budgets**.
5. Construir dashboards Grafana y alertas accionables.

## 3. Alcance

- **Incluye:** la app (RAG + Pegasus) y su interacción con la infra; dashboards y alertas.
- **Excluye:** re-arquitectura de la infra; tracing del interior de los contenedores de terceros.

## 4. Metodología

1. **Instrumentación OTel** en los puntos clave del pipeline (`intent_router`, `search`, `filtering`, `response`, guards) y en las herramientas del agente.
2. **Métricas** vía Prometheus (RED: Rate, Errors, Duration + métricas de costo/tokens).
3. **Correlación logs-trazas** con `trace_id` propagado a Graylog.
4. **Definición de SLOs** con el equipo/cátedra (p. ej. 99% de requests < X s; disponibilidad ≥ Y%).
5. **Dashboards + alertas** y prueba de que disparan ante incidentes reales (se cruza con la propuesta 01).

## 5. Stack y herramientas

- **OpenTelemetry SDK** (Python) + **OTel Collector** (`unsl-tesis-otel`).
- **Tempo** (trazas), **Prometheus** (métricas), **Grafana** (dashboards), **Graylog/OpenSearch** (logs).

## 6. Métricas / KPIs

- Latencia **por etapa** del pipeline (p50/p95/p99).
- **Tasa de error** por endpoint y por dependencia.
- **Tokens y costo** por request y agregados.
- **Hit rate** de caché (Redis).
- Cumplimiento de **SLOs** y consumo de **error budget**.

## 7. Entregables

- App instrumentada con OTel (PRs).
- Conjunto de dashboards Grafana + reglas de alerta.
- Documento de **SLIs/SLOs y error budget**.
- Guía de "cómo leer la salud del sistema".

## 8. Cronograma (orientativo)

| Semanas | Hito |
|---------|------|
| 1–3 | Instrumentación OTel (trazas) |
| 4–6 | Métricas Prometheus + costo/tokens |
| 7–8 | Correlación logs-trazas (Graylog) |
| 9–10 | Definición de SLIs/SLOs |
| 11–12 | Dashboards + alertas |
| 13–14 | Validación con incidentes + informe |

## 9. Criterios de éxito

- Una request se puede seguir de punta a punta en una sola traza.
- SLOs definidos y visibles en Grafana con alertas que disparan correctamente.
- Costo por consulta medido y visible.

## 10. Riesgos y consideraciones éticas

- No loggear datos sensibles de usuarios/consultas en claro (PII) — sanear.
- Overhead de instrumentación: medirlo y mantenerlo acotado.
