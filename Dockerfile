# ==============================================================================
# FuegoRD // IGNIS — Dockerfile para FastAPI + Uvicorn
# Backend ultra ligero en Python para Detección Satelital NASA FIRMS y Predicción IA
# ==============================================================================

FROM python:3.11-slim

# Evitar prompts interactivos y optimizar Python en contenedor
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Instalar dependencias mínimas de sistema requeridas para geoespacial y healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero para aprovechar caché de Docker
COPY requirements.txt .

# Instalar dependencias Python de forma ultra-ligera (sin PyTorch innecesario)
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar módulos del sistema
COPY config.py server.py map_service.py ./
COPY pipeline/ ./pipeline/
COPY static/ ./static/
COPY data/ ./data/

# Ejecutar con usuario no-root (seguridad de contenedor)
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exponer el puerto del servidor HTTP
EXPOSE 8080

# Healthcheck nativo
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Comando de arranque del servicio de alto rendimiento
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
