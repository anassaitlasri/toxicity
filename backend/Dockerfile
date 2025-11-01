# ---------- Base (runtime) ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Deps layer (cache-friendly) ----------
FROM base AS deps
COPY ./requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# ---------- App layer ----------
FROM base AS app
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY ./app /app/app
COPY ./uvicorn_app.py /app/uvicorn_app.py

ENV MODEL_CAMEMBERT_PATH=/models/camembert-toxicity \
    MODEL_GPT2_PATH=/models/gpt2-toxicity \
    DEFAULT_MODEL=camembert \
    CORS_ORIGINS=http://localhost:3000

VOLUME ["/models"]

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["python", "uvicorn_app.py"]
