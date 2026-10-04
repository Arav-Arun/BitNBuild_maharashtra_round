UV ?= uv
EXTRAS = --extra dev --extra ml --extra server
RUN = $(UV) run --locked $(EXTRAS)

.PHONY: help setup check lint format test build db-init dev-api dev-web web-install web-check tripcrew tripcrew-demo
.PHONY: hoprag-data hoprag forge forge-natural forge-freeze eval diagnose verify-eval regression
.PHONY: web-build

help:
	@echo "setup   Install the locked development environment"
	@echo "check   Run lint, formatting checks, and tests"
	@echo "format  Format Python and sort imports"
	@echo "build   Build wheel and source distribution in dist/"
	@echo "db-init Initialize the configured SQLite database"
	@echo "dev-api Run the API module (available after Task 10)"
	@echo "dev-web Run the Next.js development server"
	@echo "tripcrew Run 20 offline TripCrew scenarios"
	@echo "tripcrew-demo Demonstrate stale-FX repair with selective replay"
	@echo "hoprag-data Download and verify the MuSiQue-Ans development dataset"
	@echo "hoprag Record 50 HopRAG questions and verify unchanged replay (offline baseline)"
	@echo "forge   Inject faults into passing AGENT=tripcrew|hoprag runs (resumes automatically)"
	@echo "forge-natural Attribute naturally failed AGENT runs with oracle fixes (test-only)"
	@echo "forge-freeze  Export AGENT labels and write DATASET_VERSION"
	@echo "eval    Train the diagnoser, run baselines/ablations/integrity checks into data/eval"

setup:
	$(UV) sync --locked $(EXTRAS)

check: lint test

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff check --select I --fix .
	$(RUN) ruff format .

test:
	$(RUN) python -m unittest discover -s tests -v

build:
	$(UV) build

db-init:
	$(RUN) python -m blackbox.store

dev-api:
	$(RUN) python -m server

web-install:
	npm --prefix web install

web-check:
	npm --prefix web run check

dev-web:
	npm --prefix web run dev

tripcrew:
	$(RUN) python -m agents.tripcrew run

tripcrew-demo:
	$(RUN) python -m agents.tripcrew demo --report data/tripcrew/demo.json

hoprag-data:
	$(RUN) python -m agents.hoprag download

hoprag:
	$(RUN) python -m agents.hoprag run --count 50 --verify-replay

AGENT ?= tripcrew

forge:
	$(RUN) python -m blackbox.forge inject --agent $(AGENT) --target 360 --concurrency 4

forge-natural:
	$(RUN) python -m blackbox.forge natural --agent $(AGENT)

forge-freeze:
	$(RUN) python -m blackbox.forge freeze --agent $(AGENT)

eval:
	$(RUN) python -m blackbox.ml eval

web-build:
	npm --prefix web run build


diagnose:
	$(RUN) python -m blackbox.explain diagnose $(RUN_ID) --verify --save --data-dir data/$(AGENT)

verify-eval:
	$(RUN) python -m blackbox.explain verify-eval --n 30 --data-dir data/$(AGENT)

regression:
	$(RUN) python -m blackbox.explain export $(FORK) --data-dir data/$(AGENT)
