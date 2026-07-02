.PHONY: up down ingest eval test api lint

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python scripts/ingest.py --tickers NVDA AMD --forms 10-K --limit 1

eval:
	python -m sightline.eval.run

test:
	pytest -q

api:
	uvicorn sightline.api.main:app --reload --port 8000

lint:
	ruff check src tests scripts
