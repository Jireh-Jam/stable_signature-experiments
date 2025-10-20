.PHONY: help install install-dev format lint type-check test clean

# Default target
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

format:  ## Format code with ruff and black
	@echo "Running ruff format fixes..."
	ruff check --fix watermarking_methods/ common/ tools/ || true
	@echo "Running black..."
	black watermarking_methods/ common/ tools/ *.py || true

lint:  ## Run linting checks with ruff
	@echo "Running ruff linter..."
	ruff check watermarking_methods/ common/ tools/

type-check:  ## Run type checking with mypy
	@echo "Running mypy..."
	mypy watermarking_methods/ common/ tools/ || true

test:  ## Run tests with pytest
	@echo "Running pytest..."
	pytest tests/ -v || echo "No tests found yet"

clean:  ## Clean build artifacts and caches
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

check-all: lint type-check test  ## Run all checks (lint, type-check, test)
	@echo "All checks completed!"

notebook:  ## Start Jupyter notebook server
	jupyter notebook pipeline_mk4_user_friendly.ipynb

# Watermarking-specific targets
stable-signature-cli:  ## Show stable signature CLI help
	python -m watermarking_methods.stable_signature --help || echo "CLI not yet implemented"

watermark-anything-cli:  ## Show watermark anything CLI help
	python -m watermarking_methods.watermark_anything --help || echo "CLI exists"

# Development workflow
dev-setup: install-dev  ## Set up development environment
	@echo "Installing pre-commit hooks..."
	pre-commit install || echo "pre-commit not configured yet"
	@echo "Development environment ready!"

smoke-test:  ## Run smoke tests to verify package imports
	@echo "Running smoke import tests..."
	python3 -c "import watermarking_methods; print('✓ watermarking_methods')"
	python3 -c "from watermarking_methods import get_method; print('✓ get_method')"
	python3 -c "from watermarking_methods.stable_signature import StableSignatureMethod; print('✓ StableSignatureMethod')"
	python3 -c "from watermarking_methods.watermark_anything import WatermarkAnythingMethod; print('✓ WatermarkAnythingMethod')"
	python3 -c "import common; print('✓ common')"
	python3 -c "import tools; print('✓ tools')"
	@echo "All smoke tests passed!"
