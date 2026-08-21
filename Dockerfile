# ============================================================
# Dockerfile - Gold Price Prediction API
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

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
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
