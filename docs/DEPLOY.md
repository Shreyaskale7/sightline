# Deploying the Sightline demo (free)

The demo is a single FastAPI app (landing page + console + `/query` + `/upload`). Everything
below stays on free tiers.

## What ships, and what doesn't

The serving image carries the corpus, because the whole point is that a citation shows you the
*actual* page. It deliberately leaves out anything the champion retrieval config doesn't use:

| Included | Size | Why |
|---|--:|---|
| `data/pages` (2,353 page images) | 758 MB | cited pages are rendered inline |
| `data/sightline.db` | 10 MB | page metadata + provenance |
| `qdrant/sightline_text_chunks` | 26 MB | the champion index |
| **Excluded** | | |
| `qdrant/sightline_visual` | 1.7 GB | implemented + measured, but **not in the champion** (`docs/RESULTS.md`) |
| `qdrant/sightline_text` | 5 MB | unchunked baseline — ablation only |
| `data/llm_cache.db`, `.env` | — | machine-local cache and secrets never enter an image |

The image also **bakes the two models in at build time** (BGE-small embedder + BGE-reranker-base
cross-encoder) so a cold container starts without depending on a model host, and does *not*
install `colpali-engine` — that would add gigabytes to serve code that never runs.

## Option A — Hugging Face Spaces (recommended, ~20 minutes)

1. Create a free account at huggingface.co → **New Space** → SDK: **Docker** → CPU basic (free).

2. Assemble the Space checkout. The Space needs a different shape from the GitHub repo (it
   carries the corpus, which is gitignored here, plus HF's README frontmatter and LFS rules),
   so there's a script for it:

   ```bash
   python scripts/prepare_space.py --out ../sightline-space
   ```

   It prints the assembled size (~794 MB) and the exact next commands.

3. Push it:

   ```bash
   cd ../sightline-space
   git init && git lfs install
   git remote add origin https://huggingface.co/spaces/<you>/sightline
   git add -A && git commit -m "Sightline v1.0" && git push -u origin main
   ```

   The first push moves ~800 MB — expect it to take a while on a home connection. The
   `Dockerfile` is picked up automatically (port 7860).

4. **Secrets** (Space → Settings → Variables and secrets) — never commit these:
   `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `ROUTER_MODEL`, `LLM_MIN_INTERVAL_S`,
   `SEC_USER_AGENT`.

5. Open the Space URL: landing page at `/`, console at `/app`.

## Option B — Render / Fly.io free tier

Same Dockerfile and same secrets; attach a disk if you'd rather mount `data/` than bake it in.

## Option C — run it locally (zero setup, best for recording a demo)

```bash
uvicorn sightline.api.main:app
```

Serves the whole app at `http://localhost:8000` with nothing to babysit — and it's faster than
a free-tier box, which matters if you're recording. A good demo path: landing → console → a
basic question (show the highlighted citation) → the comparison question → the one it refuses →
the trace strip → upload a PDF and ask about it.

## Test the image locally first (optional but recommended)

Requires Docker Desktop to be **running**:

```bash
docker build -t sightline:v1 .
```
```bash
docker run --rm -p 7860:7860 --env-file .env sightline:v1
```

Then open `http://localhost:7860`. Catching a build problem here is much cheaper than catching
it after an 800 MB push.

## Before you demo

- **Swap the answer model.** The free nemotron is the measured weak link — five of six false
  abstentions were cases where retrieval had already found the right page. Point `LLM_MODEL` at
  a stronger model (e.g. Gemini Flash) in `.env` / the Space secrets.
- **Clear the response cache** (`rm data/llm_cache.db`) so the demo doesn't replay answers from
  the previous model.

## Notes

- Free-tier CPU boxes are slower than a laptop: expect ~10–20 s per query. Model load is
  amortized by the startup warmup, and the response cache makes repeat questions instant.
- The LLM free-tier daily quota (~50 requests) is the real limit for a public demo — fine for a
  small audience, not for traffic. The cache absorbs repeats.
- Serving holds one shared retriever and serializes queries under a lock (embedded Qdrant is
  single-client-per-path, and the CPU models aren't built for concurrency). Correct for a demo;
  a real multi-user deployment needs per-user namespaces and a concurrency model first.
- Budget guardrail: the stack points at free models. If you ever switch `LLM_MODEL` to a paid
  one, set a hard spending cap with the provider first.
