# Makefile for Watermarking Methods project

.PHONY: help install install-dev format lint type test clean notebook docs

# Default target
help:
	@echo "🔐 Watermarking Methods - Available Commands:"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  install      - Install package in editable mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo ""
	@echo "🛠️  Code Quality:"
	@echo "  format       - Format code with black and ruff"
	@echo "  lint         - Run linting with ruff"
	@echo "  type         - Run type checking with mypy"
	@echo "  test         - Run tests with pytest"
	@echo ""
	@echo "📓 Notebooks & Docs:"
	@echo "  notebook     - Start Jupyter notebook server"
	@echo "  docs         - Build documentation"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean        - Clean build artifacts and cache"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install-dev && make notebook"

# Installation targets
install:
	@echo "📦 Installing watermarking methods package..."
	pip install -e .

install-dev:
	@echo "📦 Installing with development dependencies..."
	pip install -e ".[dev,notebooks]"

# Code quality targets
format:
	@echo "🎨 Formatting code..."
	ruff --fix watermarking_methods/ || true
	black watermarking_methods/
	@echo "✅ Code formatting complete"

lint:
	@echo "🔍 Running linting..."
	ruff check watermarking_methods/
	@echo "✅ Linting complete"

type:
	@echo "🔍 Running type checking..."
	mypy watermarking_methods/
	@echo "✅ Type checking complete"

test:
	@echo "🧪 Running tests..."
	pytest -v
	@echo "✅ Tests complete"

# Development targets
notebook:
	@echo "📓 Starting Jupyter notebook server..."
	@echo "🌟 Open pipeline_mk4_user_friendly.ipynb to get started"
	jupyter notebook

docs:
	@echo "📚 Building documentation..."
	@echo "⚠️  Documentation build not yet implemented"

# Maintenance targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

# Quality gate - run all checks
check: lint type test
	@echo "✅ All quality checks passed"

# CI/CD target
ci: install-dev check
	@echo "✅ CI pipeline complete"