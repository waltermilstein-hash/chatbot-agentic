# =============================================================================
# MinIO custom (Creación de @Raven) — los repos oficiales minio/minio y dl.min.io
# ya no sirven imagen/binario público (410/denied). Se baja el binario desde
# GitHub Releases e incluye también `mc` (para el bootstrap del bucket).
# =============================================================================
FROM debian:12-slim

ARG MINIO_REL=RELEASE.2025-09-07T16-13-09Z
ARG MC_REL=RELEASE.2025-08-13T08-35-41Z
ARG MINIO_URL=https://github.com/minio/minio/releases/download/${MINIO_REL}/minio.linux-amd64.${MINIO_REL}
ARG MC_URL=https://github.com/minio/mc/releases/download/${MC_REL}/mc.linux-amd64.${MC_REL}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL "${MINIO_URL}" -o /usr/bin/minio \
    && curl -fsSL "${MC_URL}"    -o /usr/bin/mc \
    && chmod +x /usr/bin/minio /usr/bin/mc \
    && mkdir -p /data \
    && apt-get purge -y curl || true \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 9000 9001
ENV MINIO_ROOT_USER=veritas_admin \
    MINIO_ROOT_PASSWORD=veritas_admin_tesis
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
