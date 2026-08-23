# ============================================================
# Dockerfile - Gold Price Prediction API
# ============================================================
# Se fija el digest de la imagen base: una etiqueta como `3.11-slim` es movil y
# se reconstruye periodicamente, de modo que dos builds del mismo commit pueden
# no ser identicos. Para actualizar, resolver el nuevo digest con
# `docker buildx imagetools inspect python:3.11-slim` y sustituirlo aqui.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser configs/ configs/
COPY --chown=appuser:appuser models/ models/

EXPOSE 8000

# La imagen solo se considera sana cuando los artefactos estan disponibles.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')" || exit 1

USER appuser
# --host 0.0.0.0 es obligatorio DENTRO del contenedor para que el puerto sea
# alcanzable desde el host; la restriccion de exposicion se aplica en el mapeo
# de puertos de docker-compose.yml, que publica solo en 127.0.0.1.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
