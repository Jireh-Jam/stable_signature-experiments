PY=python
PIP=python -m pip

.PHONY: install dev format lint type test smoke

install:
	$(PIP) install -U pip
	$(PIP) install -e .

dev:
	$(PIP) install -U pip
	$(PIP) install -e .[torch]
	$(PIP) install ruff black mypy pytest nbformat nbconvert

format:
	ruff --fix . || true
	black .

lint:
	ruff check .

type:
	mypy stable_signature_experiments

smoke:
	$(PY) -c "import importlib; importlib.import_module('stable_signature_experiments.watermarking_methods.stable_signature'); importlib.import_module('stable_signature_experiments.watermarking_methods.watermark_anything'); print('imports: OK')"

test:
	pytest -q
