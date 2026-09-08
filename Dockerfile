# SatQuery AI — FastAPI inference service (serve/api.py)
#
# CPU-only. The site (web/) deploys to Vercel; this image is the long-lived
# backend it talks to. Build context is the filesystem, so models/weights/*
# is included even though git ignores it — see .dockerignore.
#
# Local:
#   docker build -t satquery-api .
#   docker run --rm -p 7860:7860 -e SATQUERY_ALLOWED_ORIGINS=https://your-site.vercel.app satquery-api
#
# Hugging Face Spaces (Docker SDK): expects port 7860 and a non-root user.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache \
    PORT=7860

WORKDIR /app

# CPU torch first, from the CPU wheel index, so the resolver never pulls the
# multi-GB CUDA build. Then everything else.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" \
 && pip install -r requirements.txt

COPY . .

# Runtime-writable dirs (uploads, fetched AOI scenes, run reports) and a
# non-root user with the uid Hugging Face Spaces expects.
RUN useradd -m -u 1000 app \
 && mkdir -p data/uploads data/aoi_fetch vault/runs .cache \
 && chown -R app:app /app
USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/health').status==200 else 1)"

# Shell form so ${PORT} expands — HF Spaces uses 7860, Render/Fly inject their own.
CMD uvicorn serve.api:app --host 0.0.0.0 --port ${PORT:-7860}
