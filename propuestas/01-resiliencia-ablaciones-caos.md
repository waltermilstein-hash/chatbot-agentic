# Propuesta 01 — Resiliencia de la infra + chatbot por ablaciones e ingeniería de caos

**Eje:** Confiabilidad / SRE · **Dificultad:** Media-Alta · **Duración:** 14–16 semanas

## 1. Contexto y motivación

El sistema depende de una infra distribuida (`unsl-tesis`): Weaviate (vectorial),
MongoDB (conversaciones), Redis (caché/estado), MinIO (objetos) y el proceso de
la app en el host. Hoy **no se conoce cuantitativamente** qué pasa con el chatbot
cuando un componente se degrada o cae: ¿responde igual?, ¿con peor calidad?,
¿falla elegantemente o se cuelga? Medirlo es condición para poder decir que la
base es "potente" y para guiar al alumno sobre dónde reforzarla.

## 2. Objetivos

**General:** caracterizar la **resiliencia** del chatbot RAG frente a fallos y
degradaciones de su infra, combinando **ablaciones controladas** (apagar/limitar
un componente a la vez) con **ingeniería de caos** (fallos aleatorios/realistas).

**Específicos:**
1. Mapear las dependencias reales de cada endpoint (RAG y Pegasus) a servicios de infra.
2. Diseñar un **catálogo de ablaciones**: Weaviate caído, Mongo lento, Redis sin memoria, MinIO inaccesible, OpenAI con timeouts/429, red con latencia/pérdida.
3. Definir el **comportamiento esperado vs. observado** para cada ablación (degradación aceptable vs. fallo duro).
4. Ejecutar experimentos de caos reproducibles y medir el impacto.
5. Proponer y validar **mitigaciones** (timeouts, reintentos, *circuit breakers*, *fallbacks*, degradación graciosa).

## 3. Alcance

- **Incluye:** los 4 datastores + dependencia del LLM + red; endpoints de chat y agente; métricas de disponibilidad, latencia y calidad.
- **Excluye:** fallos del hardware físico del host; escalado horizontal multi-nodo; ataques de seguridad (ver propuestas 03/04).

## 4. Metodología

1. **Modelo de dependencias:** instrumentar y trazar una petición completa (con OTel/Tempo ya disponibles) para saber qué toca cada endpoint.
2. **Ablaciones (una variable por vez):**
   - *Hard kill:* `docker stop unsl-tesis-weaviate`, etc.
   - *Degradación:* limitar CPU/mem (`--cpus`, `--memory`), inyectar latencia y pérdida de paquetes (`tc`/`netem`, Toxiproxy o Pumba).
   - *Dependencia externa:* simular timeouts/429 de OpenAI con un proxy mock.
3. **Ingeniería de caos:** experimentos con hipótesis previa (steady-state → inyección → observación → aprendizaje), estilo Principles of Chaos. Herramientas candidatas: **Pumba** (caos para contenedores Docker, encaja con el stack), **Toxiproxy** (fallos de red por servicio), scripts propios.
4. **Carga controlada:** generar tráfico reproducible (ver stack en propuesta 06) durante cada experimento.
5. **Análisis:** comparar *steady state* vs. estado inyectado; clasificar cada resultado (resiliente / degradado / fallo).

## 5. Stack y herramientas

- Docker / Docker Compose (control de los contenedores `unsl-tesis-*`).
- **Pumba** y/o **Toxiproxy**, `tc netem` para caos de red.
- **OTel Collector + Tempo + Prometheus + Grafana** (ya en la infra) para medir.
- Script de carga (k6 / Locust) para tráfico de fondo.

## 6. Métricas / KPIs

- **Disponibilidad** del endpoint bajo fallo (% de respuestas exitosas).
- **Latencia** p50/p95/p99 en estado normal vs. degradado.
- **Calidad** de la respuesta bajo degradación (reutilizar las métricas de la propuesta 02).
- **MTTR** / tiempo de recuperación tras restaurar el componente.
- **Tasa de fallos duros vs. degradación graciosa**.

## 7. Entregables

- Catálogo de ablaciones y matriz **componente × impacto**.
- Suite de experimentos de caos **reproducible** (scripts + README).
- Dashboard Grafana de resiliencia.
- Informe con hallazgos + **mitigaciones implementadas y validadas** (PRs al repo).

## 8. Cronograma (orientativo)

| Semanas | Hito |
|---------|------|
| 1–3 | Modelo de dependencias + instrumentación/trazas |
| 4–6 | Catálogo de ablaciones + baseline de *steady state* |
| 7–10 | Experimentos de caos y medición |
| 11–13 | Mitigaciones (timeouts/retries/circuit breakers/fallbacks) + revalidación |
| 14–16 | Dashboard, informe y defensa |

## 9. Criterios de éxito

- Toda ablación tiene comportamiento esperado documentado y verificado.
- Al menos 3 mitigaciones implementadas que mejoren medidamente la resiliencia.
- Experimentos re-ejecutables por un tercero con un comando.

## 10. Riesgos y consideraciones éticas

- Ejecutar **solo** sobre la infra local `unsl-tesis` (loopback), nunca sobre servicios compartidos (ej. el stack vecino `molto-uat`, que es de otro proyecto).
- El caos puede corromper datos de prueba: trabajar con volúmenes descartables / backups.
- Cuidar el costo de la API OpenAI durante las pruebas de carga (usar mocks cuando se pueda).
