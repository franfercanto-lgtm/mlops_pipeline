# Misma versión de Python que se usó para entrenar el modelo
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Usuario sin privilegios (no se corre como root)
RUN useradd --create-home --uid 1000 appuser

# 1) Dependencias primero: esta capa se cachea mientras no cambie requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Código de la API + modelo entrenado (los 2 joblibs)
COPY custom_transformers.py schemas.py main.py ./
COPY artifacts/ ./artifacts/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
