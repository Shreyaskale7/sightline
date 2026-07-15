# Deploying the Sightline demo (free)

The demo is a single FastAPI app (landing page + console + `/query`). Everything below stays
on free tiers.

## Option A — Hugging Face Spaces (recommended, ~15 minutes)

1. Create a free account at huggingface.co, then **New Space** → SDK: **Docker** → CPU basic
   (free).
2. Push this repo to the Space (HF gives you a git remote):
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/sightline
   git push space main
   ```
   The `Dockerfile` in the repo root is picked up automatically (port 7860).
3. **Secrets** (Space → Settings → Variables and secrets): add `LLM_API_KEY`, `LLM_BASE_URL`,
   `LLM_MODEL`, `ROUTER_MODEL`, `LLM_MIN_INTERVAL_S`, `SEC_USER_AGENT` — same values as your
   local `.env`. Never commit `.env` itself.
4. **The corpus**: `data/` is gitignored (≈2 GB of page images). Two choices:
   - *Small demo corpus (simplest)*: ingest a subset in the Space's container once —
     `python scripts/ingest.py -t NVDA --forms 10-K --limit 1 && python scripts/index.py --chunked`
     (add a persistent storage volume in Space settings first, or re-run on each restart).
   - *Full corpus*: upload `data/` to the Space's persistent storage (Settings → storage),
     or host it as a HF Dataset and download at startup.
5. Open the Space URL — the landing page is `/`, the console `/app`.

## Option B — Render / Fly.io free tier

Same Dockerfile. Set the env vars in the dashboard, attach a disk for `data/`, done.

## Option C — run it locally (zero setup)

`uvicorn sightline.api.main:app` serves the whole app at `http://localhost:8000` with nothing
to babysit — useful for a quick demo: landing page → console → a basic question (show the
highlighted citation) → the comparison question → the question it refuses → the trace strip.

## Notes

- Free-tier CPU boxes are slower than your laptop: expect ~10–20 s per query (model load is
  amortized by the startup warmup; the response cache makes repeated demo questions instant).
- The LLM key's free-tier daily quota (~50 requests) is the real limit for a public demo —
  fine for a small audience, not for high traffic. The cache absorbs repeat questions.
- Budget guardrail: the stack points at free models; if you ever switch `LLM_MODEL` to a paid
  one, set a hard spending cap with the provider first.
