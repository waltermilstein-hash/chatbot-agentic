# Propuesta 04 — Red-team de guardrails y evaluación del pipeline de compliance

**Eje:** Seguridad / Gobernanza de IA · **Dificultad:** Alta · **Duración:** 14–16 semanas

> ⚖️ Trabajo **defensivo** sobre el sistema propio. Complementa la propuesta 03:
> allí se atacan las entradas; aquí se **evalúa y endurece la capa de control**.

## 1. Contexto y motivación

El pipeline RAG ya incluye una capa de *compliance* notable:
`ai_act_guard`, `vera_multidimensional`, `bvg`, `contradiction_detector` y
`audit_trail` (en `chatbot/pipeline_v2/compliance/`). Esa capa es un
diferencial del proyecto, pero **no está medida**: no se sabe su tasa de
detección, sus falsos positivos ni si se puede evadir. Esta propuesta la convierte
en algo **evaluado y confiable**, alineado con el marco del **EU AI Act**.

## 2. Objetivos

**General:** **medir y mejorar** la eficacia de los guardrails y del pipeline de
compliance frente a entradas adversarias y contenido problemático.

**Específicos:**
1. Documentar qué detecta/bloquea hoy cada guard (AI-Act, VERA, BVG, contradicciones) y cómo registra el `audit_trail`.
2. Construir un **dataset de evaluación de seguridad de contenido** (tóxico, PII, sesgo, contradicciones, usos prohibidos según categorías de riesgo del AI Act) + contraejemplos benignos.
3. Medir **detección vs. falsos positivos** de cada guard.
4. **Red-team de evasión:** intentar que contenido problemático pase los guards (paráfrasis, codificación, cambio de idioma, *obfuscation*).
5. Proponer mejoras y verificar la **trazabilidad/auditoría** (¿el `audit_trail` permite reconstruir qué pasó y por qué?).

## 3. Alcance

- **Incluye:** los guards del pipeline RAG, el audit trail, y su alineación conceptual con el AI Act.
- **Excluye:** certificación legal formal (es un trabajo de ingeniería, no un dictamen jurídico); el agente Pegasus (parcialmente, salvo que comparta guards).

## 4. Metodología

1. **Inventario de controles:** leer el código de cada guard y documentar entradas, criterios y salidas.
2. **Dataset etiquetado:** combinar corpus públicos de toxicidad/PII/sesgo + casos propios del dominio legal; etiquetar "debe bloquear" / "debe permitir".
3. **Evaluación base:** matriz de confusión por guard (TP/FP/TN/FN), precisión/recall.
4. **Red-team de evasión:** aplicar transformaciones adversarias y medir cuántas evaden la detección.
5. **Mejoras + revalidación**, cuidando no aumentar falsos positivos (cruce con propuesta 02).
6. **Auditoría:** verificar que el `audit_trail` sea completo, inmutable y reconstruya la decisión.

## 5. Stack y herramientas

- Corpus de contenido (p. ej. datasets de toxicidad/PII), métricas de clasificación.
- **Graylog/OpenSearch** (en la infra) para analizar el audit trail.
- MLflow para registrar corridas; framework de guardrails como referencia (NeMo Guardrails, Llama Guard) para comparar.

## 6. Métricas / KPIs

- **Precisión/Recall/F1** por guard y global.
- **Tasa de evasión** bajo ataque adversario.
- **Tasa de falsos positivos** sobre consultas legítimas.
- **Cobertura del audit trail** (¿toda decisión queda registrada y es reconstruible?).

## 7. Entregables

- Documento de inventario de controles y su mapeo al AI Act.
- Dataset de evaluación de seguridad de contenido (versionado).
- Informe con matrices de confusión, tasa de evasión y recomendaciones.
- Mejoras implementadas (PRs) + regresión en CI.

## 8. Cronograma (orientativo)

| Semanas | Hito |
|---------|------|
| 1–3 | Inventario de controles + mapeo AI Act |
| 4–6 | Dataset etiquetado |
| 7–9 | Evaluación base (confusión por guard) |
| 10–12 | Red-team de evasión |
| 13–14 | Mejoras + auditoría + revalidación |
| 15–16 | Informe y defensa |

## 9. Criterios de éxito

- Cada guard tiene métricas de detección y FP documentadas.
- Tasa de evasión medida y reducida tras las mejoras.
- Audit trail verificado como completo y reconstruible.

## 10. Riesgos y consideraciones éticas

- Manejo cuidadoso de datasets con contenido sensible (tóxico/PII): acceso restringido, no redistribuir.
- El mapeo al AI Act es orientativo (ingeniería), no asesoramiento legal.
- Balancear seguridad y utilidad: una suba de falsos positivos degrada el servicio.
