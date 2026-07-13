# Sightline demo image — serves the landing page, console, and /query API.
#
# Built for free-tier hosting (Hugging Face Spaces / Render / Fly):
#   docker build -t sightline .
#   docker run -p 7860:7860 -v ./data:/app/data --env-file .env sightline
#
# The corpus (data/) is NOT baked into the image — mount it, or run the ingest +
# index scripts once inside the container (see docs/DEPLOY.md). HF Spaces uses
# port 7860 by convention.
FROM python:3.11-slim

WORKDIR /app

# Layer-cache dependencies separately from source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[retrieval]" \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir colpali-engine sentence-transformers

COPY scripts ./scripts

# Models download on first use into this cache; mount it to persist across restarts.
ENV HF_HOME=/app/.hf-cache
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn sightline.api.main:app --host 0.0.0.0 --port ${PORT}"]
