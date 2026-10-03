UV ?= uv
RUN = $(UV) run --locked --extra dev

.PHONY: help setup check lint format test smoke sample build

help:
	@echo "setup   Install the locked development environment"
	@echo "check   Run lint, formatting checks, tests, and CLI smoke check"
	@echo "format  Format Python and sort imports"
	@echo "sample  Write a sample trace to blackbox/data"
	@echo "build   Build wheel and source distribution in dist/"

setup:
	$(UV) sync --locked --extra dev

check: lint test smoke

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff check --select I --fix .
	$(RUN) ruff format .

test:
	$(RUN) python -m unittest discover -s tests -v

smoke:
	$(RUN) python scripts/smoke.py

sample:
	$(RUN) blackbox sample

build:
	$(UV) build
