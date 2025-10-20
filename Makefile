PY = python3
PKG = stable_signature_experiments

.PHONY: install dev format lint type test smoke-notebook

install:
	$(PY) -m pip install -U pip
	pip install -e .

dev:
	pip install -e .[dev]

format:
	ruff --fix . || true
	black .

lint:
	ruff check .

type:
	mypy $(PKG)

smoke:
	$(PY) -c "import stable_signature_experiments.watermarking_methods as wm; print('import OK')"

smoke-notebook:
	$(PY) - <<'PY'
import nbformat
nb = nbformat.read('pipeline_mk4_user_friendly.ipynb', as_version=4)
print('notebook loaded: OK')
PY
