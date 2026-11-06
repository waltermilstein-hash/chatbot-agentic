# Propuesta 02 — Evaluación sistemática de la calidad del RAG

**Eje:** Calidad / ML · **Dificultad:** Media · **Duración:** 12–14 semanas

## 1. Contexto y motivación

El motor RAG (`chatbot/pipeline_v2/`) hace recuperación vectorial, multi-query,
filtrado de *chunks* y respuesta con citas. Sin un marco de evaluación, no se
puede afirmar que un cambio (otro ranker, otro *chunking*, otro modelo) **mejora**
o **empeora** las respuestas. Esta propuesta construye el "termómetro" de calidad
que habilita a todas las demás (ej. sirve para medir calidad bajo caos en la 01).

## 2. Objetivos

**General:** diseñar e implementar un marco reproducible para medir la calidad
del pipeline RAG a nivel de **recuperación** y de **generación**.

**Específicos:**
1. Construir un **dataset de evaluación** (preguntas + respuestas/pasajes de referencia) sobre los documentos de la cátedra.
2. Medir **retrieval** (hit rate, MRR, nDCG, recall@k) sobre Weaviate.
3. Medir **generación** (faithfulness/fidelidad a las fuentes, relevancia, exactitud de citas, tasa de alucinación).
4. Montar un pipeline de evaluación automático (CI) que corra ante cada cambio.
5. Realizar **ablaciones del propio RAG**: con/sin multi-query, distintos `k`, con/sin re-ranking, distintos modos de respuesta.

## 3. Alcance

- **Incluye:** endpoint de chat RAG, componentes de `pipeline_v2` (search/filtering/response), citas.
- **Excluye:** el agente Pegasus (web), que tiene su propia naturaleza; evaluación humana a gran escala (se usa muestra).

## 4. Metodología

1. **Dataset:** curar un *golden set* (50–150 preguntas) con respuestas y pasajes de referencia; incluir casos difíciles (multi-hop, "no está en los documentos", contradicciones).
2. **Métricas de retrieval:** recall@k, MRR, nDCG comparando *chunks* recuperados vs. pasajes de referencia.
3. **Métricas de generación:** usar un marco tipo **RAGAS** / **LLM-as-judge** (faithfulness, answer relevancy, context precision/recall) + verificación de citas.
4. **Ablaciones del pipeline:** variar un hiperparámetro por vez y medir el delta.
5. **Reproducibilidad:** versionar dataset y resultados; registrar corridas en **MLflow** (ya está en la infra).

## 5. Stack y herramientas

- RAGAS / DeepEval (o LLM-as-judge propio con Claude/GPT).
- **MLflow** (`unsl-tesis-mlflow`) para tracking de experimentos.
- Weaviate para las métricas de retrieval.
- Notebooks + scripts de evaluación en CI.

## 6. Métricas / KPIs

- Retrieval: **recall@k, MRR, nDCG**.
- Generación: **faithfulness, answer relevancy, context precision/recall**.
- **Tasa de alucinación** y **exactitud de citas**.
- **Abstención correcta** ("no sé" cuando la respuesta no está en el corpus).

## 7. Entregables

- *Golden set* versionado + guía de cómo ampliarlo.
- Script/CI de evaluación con reporte automático.
- Tabla de ablaciones con el impacto de cada componente del RAG.
- Recomendación fundada de la mejor configuración.

## 8. Cronograma (orientativo)

| Semanas | Hito |
|---------|------|
| 1–3 | Construcción del *golden set* |
| 4–6 | Métricas de retrieval + baseline |
| 7–9 | Métricas de generación (faithfulness/citas) |
| 10–12 | Ablaciones del pipeline + MLflow |
| 13–14 | Informe y defensa |

## 9. Criterios de éxito

- Marco de evaluación ejecutable con un comando y resultados reproducibles.
- Cada componente del RAG tiene su aporte cuantificado.
- Identificación de al menos una mejora de configuración validada.

## 10. Riesgos y consideraciones éticas

- Costo de API del *judge*: cachear y usar muestras.
- Sesgo del LLM-as-judge: validar contra una submuestra etiquetada por humanos.
- Datos de la cátedra: respetar su confidencialidad; no exfiltrar a servicios externos sin autorización.
