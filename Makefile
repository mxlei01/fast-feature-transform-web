# Quality gates: the same targets locally and in CI.
lint:
	uv run ruff check app training
ty:
	uv run ty check app training
format:
	uv run ruff format app training
	uv run ruff check --fix app training
format-check:
	uv run ruff format --check app training
pytest:
	uv run pytest -v
bandit:
	uv run bandit -r -c pyproject.toml app training
all: format lint ty pytest bandit

.PHONY: lint ty format format-check pytest bandit all

# The model: download a month of TLC trips into data/ and write artifacts/.
MONTH ?= 2026-08
train:
	uv run python -m training.train --month $(MONTH)

# The site. `run` is for development: DEBUG on, and the dev server without its autoreloader, which
# races on imports under free-threaded Python. `serve` runs gunicorn as a deployment does: one process,
# so one copy of the model and one rate-limit budget per client, with threads that run in parallel on
# free-threaded Python. It needs DJANGO_SECRET_KEY and, behind no TLS proxy, DJANGO_SSL_REDIRECT=false.
run:
	cd app && DJANGO_DEBUG=true uv run python manage.py runserver --noreload
serve:
	cd app && uv run gunicorn --bind 127.0.0.1:8000 --workers 1 --threads 8 --worker-class gthread --timeout 30 app.wsgi:application

.PHONY: train run serve
