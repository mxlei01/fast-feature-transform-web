# The fare site: Django under gunicorn, with the trained model baked in. Build from the repository
# root:
#   docker build -t fare-web .
# uv installs free-threaded CPython 3.14 and the locked dependencies (torch's CPU-only build on
# Linux). The build cannot succeed yet: the vendored fast-inference wheel is macOS-only, and the image
# needs the library's Linux wheel from PyPI first (deploy/README.md).
FROM ghcr.io/astral-sh/uv:0.12.17-trixie-slim@sha256:1c97a153d9c2308c89ca4fb19766483db3256a86313806fed99b1c59a9767b6e

# Python goes under /opt, where the non-root user can read it.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=app.settings \
    HOME=/tmp

WORKDIR /app

# Dependency layer, rebuilt only when the lock or the wheel changes. No dev group, and no uv cache
# left behind in the image.
COPY pyproject.toml uv.lock .python-version ./
COPY wheels ./wheels
RUN uv sync --frozen --no-install-project --no-default-groups --no-cache

# The site, the feature code the served model imports, and the model itself.
COPY app ./app
COPY training ./training
COPY artifacts ./artifacts
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin fare
USER 1000:1000

WORKDIR /app/app
EXPOSE 8000
# One process, so one copy of the model and one rate-limit budget per client; eight threads, which
# run in parallel on free-threaded Python.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--worker-class", "gthread", "--timeout", "30", "app.wsgi:application"]
