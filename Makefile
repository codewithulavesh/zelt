.PHONY: install test lint build clean serve ui docker

install:
	pip install -e ".[full]"
	pip install pytest twine build

test:
	pytest tests/ -v --durations=5

lint:
	ruff check .
	ruff format --check .

build: clean
	python -m build
	twine check dist/*

serve:
	zelt serve --host 0.0.0.0 --port 8000

ui:
	zelt ui --port 8000

docker:
	docker build -t zelt-ai:latest .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
