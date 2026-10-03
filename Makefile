UV ?= uv
RUN = $(UV) run --locked --extra dev

.PHONY: help setup check lint format test build db-init dev-api dev-web web-install web-check tripcrew tripcrew-demo

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

setup:
	$(UV) sync --locked --extra dev

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
