# LongVideoEditAgent — common developer tasks.
# Run `make help` to list targets.

SHELL := /usr/bin/env bash
PYTHON ?= python
PYTHONPATH_PREFIX := PYTHONPATH=src

.PHONY: help install install-dev test test-unit test-integration lint clean demo viz fixture

help: ## List available make targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Editable install of the base package (v0.1 mock pipeline)
	$(PYTHON) -m pip install -e .

install-dev: ## Editable install + every optional extra (real backends + dev tools)
	$(PYTHON) -m pip install -e '.[all]'

test: ## Run the full pytest suite
	$(PYTHONPATH_PREFIX) $(PYTHON) -m pytest tests/ -q

test-unit: ## Run only unit tests
	$(PYTHONPATH_PREFIX) $(PYTHON) -m pytest tests/unit/ -q

test-integration: ## Run only integration tests
	$(PYTHONPATH_PREFIX) $(PYTHON) -m pytest tests/integration/ -q

lint: ## Lint with pyflakes
	$(PYTHONPATH_PREFIX) $(PYTHON) -m pyflakes src/ tests/ benchmark/ scripts/

fixture: ## (Re)generate tests/fixtures/tiny_clip.mp4
	$(PYTHONPATH_PREFIX) $(PYTHON) -c "from longvideoagent.utils.video_io import write_silent_color_clip; \
		write_silent_color_clip('tests/fixtures/tiny_clip.mp4', duration_s=4.0, fps=24, \
		                       width=160, height=120, color=(50,120,200))"

demo: fixture ## End-to-end demo run on tests/fixtures/tiny_clip.mp4 → outputs/demo.mp4
	rm -rf .cache/demo
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/run_pipeline.py \
		--source tests/fixtures/tiny_clip.mp4 \
		--cache-dir .cache/demo \
		--user-prompt "Make a 4-second highlight reel with high energy" \
		--output outputs/demo.mp4 \
		--trajectory-log outputs/demo_trajectory.jsonl

viz: ## Pretty-print the trajectory log from `make demo`
	$(PYTHONPATH_PREFIX) $(PYTHON) scripts/visualize_trajectory.py --log outputs/demo_trajectory.jsonl

clean: ## Remove caches, build artifacts, demo outputs
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	rm -rf outputs/demo.mp4 outputs/demo_trajectory.jsonl outputs/generated
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
