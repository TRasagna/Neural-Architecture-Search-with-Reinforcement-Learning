# NAS-RL Makefile
.PHONY: help install test

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  dev-search   Run development search"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

dev-search:
	python -m src.ray.run_search --config configs/dev.yaml
