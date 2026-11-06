# Universidad de San Luis · Tesis — Infraestructura (datos, eventos, logs, telemetría, pentest)

Solo **contenedores de infraestructura**. El código Python (chatbot + agente)
corre **fuera de Docker**, en el host, y se conecta por la red sintética
**127.0.0.7**. Esta versión **no** sincroniza con SharePoint: los datos se cargan
**a mano por el endpoint**.

Proyecto compose: `unsl-tesis` · red: `unsl-tesis-net` · cada contenedor lleva
`-tesis-` en el nombre · todo publicado en **127.0.0.7**.

## Arquitectura claim-check

Los eventos **no** llevan el archivo: **Kafka transporta solo la referencia**
(bucket + clave) y **el archivo vive en MinIO**. El consumidor lee la referencia
del evento y descarga el payload de MinIO. El one-shot `minio-setup` deja creado
el bucket (`MINIO_BUCKET`, por defecto `unsl-documentos`).

```
   productor ──(1) sube archivo──▶ MinIO (bucket unsl-documentos)
       │
       └──(2) publica referencia {bucket,key}──▶ Kafka (topic)
                                                   │
   consumidor ◀──(3) lee referencia───────────────┘
       └──(4) descarga payload──▶ MinIO
```

## Componentes

### Datos / eventos / logs — arrancan por defecto
| Servicio | Contenedor | Endpoint (127.0.0.7) | Rol |
|----------|------------|----------------------|-----|
| MongoDB | `unsl-tesis-mongo` | `27017` | documentos / metadatos |
| Redis | `unsl-tesis-redis` | `6379` | caché / estado |
| Weaviate | `unsl-tesis-weaviate` | `8087` | base vectorial (RAG) |
| Kafka (KRaft) | `unsl-tesis-kafka` | `9092` | eventos (referencia claim-check) |
| MinIO | `unsl-tesis-minio` | `9000` · consola `9001` | objetos (payload claim-check) |
| MinIO setup | `unsl-tesis-minio-setup` | — | one-shot: crea el bucket |
| OpenSearch | `unsl-tesis-opensearch` | (interno) | motor de Graylog |
| Graylog | `unsl-tesis-graylog` | `9900` · GELF `12201` · syslog `1514` | gestión de logs |

### Telemetría / observabilidad — perfil `obs`
| Servicio | Contenedor | Endpoint (127.0.0.7) | Rol |
|----------|------------|----------------------|-----|
| Prometheus | `unsl-tesis-prometheus` | `9090` | métricas |
| Grafana | `unsl-tesis-grafana` | `3000` | dashboards |
| Tempo | `unsl-tesis-tempo` | (interno `3200`) | trazas |
| OTel Collector | `unsl-tesis-otel` | `4317` gRPC · `4318` HTTP | ingesta OTLP |
| MLflow | `unsl-tesis-mlflow` | `5000` | tracking de experimentos |

### Reverse proxy — perfil `proxy`
| Servicio | Contenedor | Endpoint (127.0.0.7) | Rol |
|----------|------------|----------------------|-----|
| Traefik | `unsl-tesis-traefik` | `80` · `443` · dashboard `8090` | proxy + TLS |

### Pentest — perfil `pentest` (a demanda, efímero)
| Servicio | Contenedor | Rol |
|----------|------------|-----|
| Pentest toolbox | `unsl-tesis-pentest` | schemathesis, nuclei, ffuf, nikto, sqlmap, ZAP headless, jwt_tool, trivy, testssl.sh… |

## Uso

```bash
cp .env.example .env

# Núcleo (datos + eventos + logs):
docker compose up -d

# Con telemetría y proxy:
docker compose --profile obs --profile proxy up -d

# Pentest (toolbox efímero, red host → ataca 127.0.0.7):
docker compose --profile pentest build pentest
docker compose --profile pentest run --rm pentest
```

Comprobaciones:

```bash
curl http://127.0.0.7:8087/v1/.well-known/ready    # weaviate listo
redis-cli -h 127.0.0.7 ping                         # PONG
# MinIO consola:  http://127.0.0.7:9001
# Graylog:        http://127.0.0.7:9900   (admin / 'admin' por defecto)
# Grafana:        http://127.0.0.7:3000   (admin / GRAFANA_ADMIN_PASSWORD)
# Prometheus:     http://127.0.0.7:9090
# MLflow:         http://127.0.0.7:5000
```

Parar / borrar:

```bash
docker compose --profile obs --profile proxy down     # parar
docker compose down -v                                 # + borrar volúmenes (datos)
```

> Nota OpenSearch: si el contenedor no arranca, el host necesita
> `sudo sysctl -w vm.max_map_count=262144`.
