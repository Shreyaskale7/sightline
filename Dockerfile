# Sightline demo image — serves the landing page, console, and /query API.
#
#   docker build -t sightline .
#   docker run -p 7860:7860 --env-file .env sightline
#
# Built for free-tier hosting (Hugging Face Spaces / Render / Fly), port 7860 by convention.
#
# Two deliberate choices, both traceable to measured results:
#   1. The visual/late-interaction stack (colpali-engine) is NOT installed. It is implemented
#      and measured (docs/RESULTS.md) but scored 0.154 on CPU and is not in the champion
#      config — installing it would add gigabytes to serve code that never runs.
#   2. Models are baked in at build time rather than downloaded on first request, so a cold
#      container starts deterministically with no network dependency on a model host.
FROM python:3.11-slim

WORKDIR /app

# Layer-cache dependencies separately from source and corpus.
COPY pyproject.toml README.md ./
COPY src ./src

# CPU-only torch: sentence-transformers needs it for the cross-encoder reranker (the single
# biggest accuracy lever). The default wheel pulls CUDA (~2 GB) that a CPU box can't use.
RUN pip install --no-cache-dir -e ".[retrieval,ocr]" \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir sentence-transformers

# Bake the two models into the image (embedder + cross-encoder reranker) so startup is
# offline and deterministic. This is the same pair the champion loads at warmup.
ENV HF_HOME=/app/.hf-cache
RUN python -c "\
from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5');\
from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base');\
print('models baked')"

COPY scripts ./scripts
# The corpus: page images + SQLite + the chunked index. See .dockerignore for what is
# deliberately left out (the visual index, the response cache, secrets).
COPY data ./data

ENV PORT=7860
EXPOSE 7860

# Serving loads the models once at startup (background warmup) — see api/main.py.
CMD ["sh", "-c", "uvicorn sightline.api.main:app --host 0.0.0.0 --port ${PORT}"]
