# Quality gates: the same targets locally and in CI.
lint:
	uv run ruff check app training deploy
ty:
	uv run ty check app training deploy
format:
	uv run ruff format app training deploy
	uv run ruff check --fix app training deploy
format-check:
	uv run ruff format --check app training deploy
pytest:
	uv run pytest -v
bandit:
	uv run bandit -r -c pyproject.toml app training deploy
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

# The container image and its deployment; see deploy/README.md. The tag is the commit the image is
# built from, so `image` refuses a tree with uncommitted changes.
IMAGE ?= fare-web
TAG := $(shell git rev-parse --short HEAD)
BRANCH ?= $(shell git branch --show-current)
image:
	@test -z "$$(git status --porcelain)" || { echo "commit first: the image tag names the commit"; exit 1; }
	docker build --platform linux/arm64 -t $(IMAGE):$(TAG) .
push:
	@test -n "$(REGISTRY)" || { echo "set REGISTRY to the cluster registry's host:port"; exit 1; }
	docker tag $(IMAGE):$(TAG) $(REGISTRY)/$(IMAGE):$(TAG)
	docker push $(REGISTRY)/$(IMAGE):$(TAG)
# Pin both overlays to this commit's image, then commit the pins on the issue branch.
pin:
	sed -i '' -E 's/^( +newTag: ).*/\1"$(TAG)"/' deploy/manifests/overlays/staging/kustomization.yaml deploy/manifests/overlays/prod/kustomization.yaml
# Point the staging Application at this branch, pushed to origin under its issue number.
deploy-staging:
	kubectl -n argocd patch application fare-web-staging --type merge -p '{"spec":{"source":{"targetRevision":"$(BRANCH)"}}}'
# Render both overlays as ArgoCD will.
manifests:
	kubectl kustomize deploy/manifests/overlays/staging > /dev/null
	kubectl kustomize deploy/manifests/overlays/prod > /dev/null
# The smoke checks against gunicorn itself, for example through kubectl port-forward:
# make smoke SMOKE_URL=http://host:port
smoke:
	FARE_BASE_URL=$(SMOKE_URL) uv run python deploy/manifests/base/smoke_check.py

.PHONY: image push pin deploy-staging manifests smoke
