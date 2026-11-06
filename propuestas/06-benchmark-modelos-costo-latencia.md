# Propuesta 06 — Benchmark de modelos: costo, latencia y calidad

**Eje:** ML Ops / Economía de la IA · **Dificultad:** Media · **Duración:** 10–12 semanas

## 1. Contexto y motivación

Hoy la app usa modelos fijos (`gpt-4o-mini` para agente y búsqueda; el del RAG
según config). No hay evidencia de que sean la mejor relación **costo / latencia
/ calidad** para el caso de la cátedra, ni una forma de decidirlo cuando aparezca
un modelo nuevo. Esta propuesta crea un banco de pruebas que convierte esa
decisión en algo medido y repetible.

## 2. Objetivos

**General:** comparar configuraciones de modelo/parametrización sobre los mismos
casos, cuantificando el trade-off costo–latencia–calidad.

**Específicos:**
1. Definir el **banco de casos** (reutiliza el *golden set* de la propuesta 02).
2. Automatizar corridas variando **modelo, temperatura, `k`, estrategia de contexto**.
3. Medir **latencia, tokens, costo** y **calidad** por configuración.
4. Incluir **generación de carga** para medir latencia bajo concurrencia.
5. Producir una **recomendación fundada** y una curva costo–calidad.

## 3. Alcance

- **Incluye:** endpoints de RAG y Pegasus; modelos accesibles vía la API configurada.
- **Excluye:** fine-tuning de modelos; despliegue de modelos locales (posible extensión).

## 4. Metodología

1. **Matriz de configuraciones** a evaluar (modelo × parámetros).
2. **Ejecución automatizada** sobre el banco de casos, registrando métricas en **MLflow**.
3. **Prueba de carga** con **k6** o **Locust** (RPS creciente) para latencia p95/p99 bajo concurrencia y detección del punto de saturación.
4. **Análisis costo–calidad:** frontera de Pareto; sensibilidad a cada parámetro.

## 5. Stack y herramientas

- **k6** / **Locust** para carga.
- **MLflow** (infra) para tracking.
- Métricas de calidad de la propuesta 02 (LLM-as-judge / RAGAS).
- Grafana/Prometheus (propuesta 05) para la foto en vivo durante la carga.

## 6. Métricas / KPIs

- **Latencia** p50/p95/p99 (incl. TTFT — *time to first token* en SSE).
- **Throughput** máximo sostenible (RPS) antes de degradar.
- **Costo por consulta** y por 1k requests.
- **Calidad** por configuración (de la propuesta 02).

## 7. Entregables

- Banco de pruebas de carga + evaluación reproducible.
- Tabla comparativa y **curva costo–calidad** (Pareto).
- Recomendación de configuración por escenario (económico / balanceado / máxima calidad).

## 8. Cronograma (orientativo)

| Semanas | Hito |
|---------|------|
| 1–2 | Matriz de configuraciones + banco de casos |
| 3–5 | Ejecución automatizada + métricas de calidad/costo |
| 6–8 | Pruebas de carga (k6/Locust) |
| 9–10 | Análisis costo–calidad (Pareto) |
| 11–12 | Informe y defensa |

## 9. Criterios de éxito

- Comparación reproducible de al menos 3–4 configuraciones.
- Recomendación respaldada por datos de costo, latencia y calidad.
- Punto de saturación del sistema identificado.

## 10. Riesgos y consideraciones éticas

- **Costo de API:** acotar el presupuesto de las corridas; cachear; usar muestras.
- No ejecutar pruebas de carga contra servicios compartidos; solo infra propia en loopback.
