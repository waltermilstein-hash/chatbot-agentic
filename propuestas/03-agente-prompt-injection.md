# Propuesta 03 — Agente de prompt injection sobre los endpoints de IA

**Eje:** Seguridad (ofensiva aplicada a defensa) · **Dificultad:** Alta · **Duración:** 14–16 semanas

> ⚖️ **Encuadre ético (leer primero).** Trabajo de **seguridad defensiva
> autorizado** sobre el **sistema propio** de la cátedra, ejecutado en
> **loopback**. El objetivo es **encontrar y cerrar** debilidades, no atacar a
> terceros. No se prueba contra servicios de otros (ej. el stack vecino
> `molto-uat`) ni contra sistemas externos. Los *payloads* y hallazgos se
> publican de forma responsable, dentro del marco académico.

## 1. Contexto y motivación

Los endpoints de IA (`/virtualbot/.../chatbotservice/stream` y
`/api/v1/agent/web...`) reciben entrada de usuario y, en el caso de Pegasus,
**leen contenido de la web** que luego el LLM procesa. Eso abre la puerta a
**prompt injection** directa (en el prompt del usuario) e **indirecta** (en una
página que el agente lee). Antes de ampliar el sistema, el alumno necesita saber
cuán expuesto está y con qué defensas cuenta.

## 2. Objetivos

**General:** construir un **agente automatizado de prueba de prompt injection**
que ejercite sistemáticamente los endpoints de IA, clasifique las respuestas y
mida la tasa de éxito de los ataques, para luego **endurecer** el sistema.

**Específicos:**
1. Catalogar las **superficies de inyección**: prompt del usuario, campos del `json_data`, documentos ingeridos (RAG), y contenido web que lee Pegasus (inyección indirecta).
2. Construir una **batería de payloads** (robo/override de system prompt, exfiltración de instrucciones, *jailbreaks*, ejecución de herramientas no deseadas en el agente ReAct, fuga de datos de otras conversaciones/tópicos).
3. Implementar un **agente evaluador** que envíe payloads, observe el stream SSE y **clasifique** éxito/fallo del ataque automáticamente.
4. Medir **tasa de éxito por categoría** y por endpoint; mapear a **OWASP LLM Top 10** (LLM01 Prompt Injection, LLM02, LLM06, etc.).
5. Diseñar, implementar y **revalidar defensas** (saneo de entrada, delimitación de contexto, *spotlighting*, validación de salida, aislamiento de instrucciones del contenido web).

## 3. Alcance

- **Incluye:** inyección directa e indirecta; el agente ReAct (uso indebido de `buscar_web`/`leer_pagina`); fuga entre tópicos/usuarios; robo de system prompt.
- **Excluye:** ataques de infraestructura (DoS, explotación de CVEs de los contenedores), que corresponden a otra línea; ingeniería social a humanos.

## 4. Metodología

1. **Modelado de amenazas** de los endpoints de IA (superficies + activos a proteger: system prompt, datos de otros usuarios, herramientas del agente).
2. **Corpus de payloads:** combinar catálogos públicos (p. ej. colecciones tipo *garak*, listas de *jailbreaks*) con payloads propios adaptados al dominio (legal/UNSL) y a Pegasus.
3. **Inyección indirecta:** montar páginas señuelo locales que Pegasus pueda leer, con instrucciones ocultas, y verificar si el agente las obedece.
4. **Agente evaluador automatizado:** orquesta envío → captura de SSE → **clasificación** del resultado (ataque exitoso / mitigado / ambiguo) con reglas + LLM-as-judge.
5. **Ciclo de endurecimiento:** implementar defensas, re-correr la batería, medir reducción de la tasa de éxito (regresión de seguridad en CI).
6. Considerar herramientas de *scanning* de LLMs existentes (**garak**, **PyRIT**) como base o comparación.

## 5. Stack y herramientas

- **garak** y/o **PyRIT** (scanners de LLM) como referencia/base.
- Cliente propio para los endpoints SSE del proyecto.
- **pentest toolbox** de la infra (`unsl-tesis-pentest`) para alojar el agente/payloads.
- LLM-as-judge para clasificar respuestas; MLflow/Graylog para registrar corridas.

## 6. Métricas / KPIs

- **Attack Success Rate (ASR)** por categoría y por endpoint.
- **Reducción de ASR** antes/después de cada defensa.
- Cobertura de la **OWASP LLM Top 10**.
- Falsos positivos de las defensas (impacto en consultas legítimas → cruzar con propuesta 02).

## 7. Entregables

- Agente evaluador de prompt injection **reproducible** (en `pentest/` o como módulo propio).
- Corpus de payloads versionado y categorizado.
- Informe de seguridad: hallazgos, severidad, mapeo OWASP y **recomendaciones**.
- Defensas implementadas (PRs) + **suite de regresión de seguridad** en CI.

## 8. Cronograma (orientativo)

| Semanas | Hito |
|---------|------|
| 1–3 | Modelado de amenazas + superficies de inyección |
| 4–6 | Corpus de payloads + cliente SSE |
| 7–9 | Agente evaluador + clasificación automática |
| 10–11 | Inyección indirecta (páginas señuelo para Pegasus) |
| 12–14 | Defensas + revalidación + regresión en CI |
| 15–16 | Informe de seguridad y defensa |

## 9. Criterios de éxito

- Batería automatizada que corre contra todos los endpoints de IA con un comando.
- ASR medido por categoría con baseline y mejora demostrada tras el endurecimiento.
- Al menos las vulnerabilidades críticas detectadas quedan mitigadas y con test de regresión.

## 10. Riesgos y consideraciones éticas

- **Solo sistema propio, en loopback.** Nada contra terceros ni la web real más allá de las páginas señuelo locales.
- Divulgación responsable de los payloads (quedan en el ámbito académico).
- Controlar costo de API; preferir modelos chicos para el *fuzzing* masivo.
- Evitar generar o almacenar contenido dañino real: los payloads son de prueba de control, no de daño.
