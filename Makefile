# Makefile for Watermarking Pipeline

.PHONY: help install dev-install format lint type-check test clean

help:
	@echo "Available commands:"
	@echo "  make install      Install the package in production mode"
	@echo "  make dev-install  Install the package in development mode with dev dependencies"
	@echo "  make format       Format code with black and ruff"
	@echo "  make lint         Run linting checks with ruff"
	@echo "  make type-check   Run type checking with mypy"
	@echo "  make test         Run tests with pytest"
	@echo "  make clean        Clean up generated files and caches"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

format:
	black stable_signature-experiments/
	ruff check --fix stable_signature-experiments/

lint:
	ruff check stable_signature-experiments/

type-check:
	mypy stable_signature-experiments/

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +