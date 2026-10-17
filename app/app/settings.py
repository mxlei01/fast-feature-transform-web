"""Django settings, all from the environment.

The site is stateless: there is no database, no session and no login, so the only cookie is
Django's CSRF token for the form. Every input arrives through ``fare.forms.TripForm``.

Environment:
    DJANGO_DEBUG: ``true`` for development; off by default.
    DJANGO_SECRET_KEY: required unless DJANGO_DEBUG is on.
    DJANGO_ALLOWED_HOSTS: comma-separated host names (default ``localhost,127.0.0.1``).
    DJANGO_CSRF_TRUSTED_ORIGINS: comma-separated origins, for a deployment behind a proxy.
    DJANGO_SSL_REDIRECT: redirect plain HTTP to HTTPS when DEBUG is off (default true).
    DJANGO_HSTS_SECONDS, DJANGO_HSTS_SUBDOMAINS, DJANGO_HSTS_PRELOAD: HSTS over HTTPS (one year; off; off).
    FARE_ARTIFACTS_DIR: the trained model (default ``artifacts/`` at the repository root).
    FARE_RATE_LIMIT: predictions per client per minute (default 30), counted per server process.
    RATELIMIT_TRUSTED_FORWARDED_HOPS: proxies in front of the site that append X-Forwarded-For (default 0).
    RATELIMIT_TRUST_CF_HEADER: take the client's address from Cloudflare's CF-Connecting-IP (default false).
    PROBE_BAN_SECONDS: how long an address that asked for a scanner path is refused (default 3600).
    LOG_FORMAT: ``json`` or ``plain`` (default plain under DEBUG, json otherwise).
    LOG_LEVEL: default INFO.
"""

import os
import secrets
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR: Path = Path(__file__).resolve().parent.parent
REPO_DIR: Path = BASE_DIR.parent


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Args:
        name: Variable name.
        default: Value when unset.

    Returns:
        True for ``true``, ``1``, ``yes`` or ``on`` (any case).

    """
    value: str | None = os.getenv(name)
    return default if value is None else value.strip().lower() in {"true", "1", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    """Read a comma-separated environment variable, dropping blanks.

    Args:
        name: Variable name.
        default: Value when unset.

    Returns:
        The stripped, non-empty entries.

    """
    return [entry.strip() for entry in os.getenv(name, default).split(",") if entry.strip()]


DEBUG: bool = _env_bool("DJANGO_DEBUG", False)
SECRET_KEY: str = os.getenv("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is off")
    # A fresh key per development process: nothing signed needs to outlive a restart, and no key is committed.
    SECRET_KEY = secrets.token_urlsafe(50)

ALLOWED_HOSTS: list[str] = _env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
CSRF_TRUSTED_ORIGINS: list[str] = _env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "")

INSTALLED_APPS: list[str] = ["django.contrib.staticfiles", "fare"]

MIDDLEWARE: list[str] = [
    # Outermost, so it times the whole stack and sees every response.
    "fare.request_log.RequestLogMiddleware",
    "django.middleware.security.SecurityMiddleware",
    # Outside everything that can answer early, so those answers carry the headers too.
    "fare.security_headers.SecurityHeadersMiddleware",
    # Before the guard, so a banned address still gets the stylesheet. The guard's 404s are no-store, which
    # keeps Cloudflare from caching them; this order covers the stylesheet even under a cache rule that
    # ignores that header, where a banned address's 404 for it would be kept and served to everyone.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    # Before anything that does work: a scanner's request, and every request from its address, ends here.
    "fare.probe_guard.ProbeGuardMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF: str = "app.urls"
WSGI_APPLICATION: str = "app.wsgi.application"
TEMPLATES: list[dict] = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    }
]

# No database: nothing here stores anything.
DATABASES: dict = {}

# Pickup times are New York wall-clock times, as in the trip records the model learned from.
USE_TZ: bool = False
TIME_ZONE: str = "America/New_York"
LANGUAGE_CODE: str = "en-us"
USE_I18N: bool = False

STATIC_URL: str = "static/"
# WhiteNoise serves the app's one stylesheet straight from fare/static, so there is no collectstatic step.
WHITENOISE_USE_FINDERS: bool = True

# Requests are tiny: a form or a JSON object of four fields.
DATA_UPLOAD_MAX_MEMORY_SIZE: int = 4096
DATA_UPLOAD_MAX_NUMBER_FIELDS: int = 16
FILE_UPLOAD_MAX_MEMORY_SIZE: int = 0
# The site takes no files. Refuse a file part before it is written: the 4 KB cap above does not count
# file data, and the CSRF check reads the whole body, writing each file to /tmp, before it rejects a
# request without a token.
DATA_UPLOAD_MAX_NUMBER_FILES: int = 0

SECURE_CONTENT_TYPE_NOSNIFF: bool = True
SECURE_REFERRER_POLICY: str = "same-origin"
X_FRAME_OPTIONS: str = "DENY"
CSRF_COOKIE_HTTPONLY: bool = True
CSRF_COOKIE_SAMESITE: str = "Strict"
if not DEBUG:
    SECURE_PROXY_SSL_HEADER: tuple[str, str] = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT: bool = _env_bool("DJANGO_SSL_REDIRECT", True)
    # A probe that talks to gunicorn directly gets the health check, not a redirect.
    SECURE_REDIRECT_EXEMPT: list[str] = [r"^healthz$"]
    # Sent only over HTTPS: a year, the usual value once TLS is in place; subdomains and preload are opt-in.
    SECURE_HSTS_SECONDS: int = int(os.getenv("DJANGO_HSTS_SECONDS", "31536000"))
    SECURE_HSTS_INCLUDE_SUBDOMAINS: bool = _env_bool("DJANGO_HSTS_SUBDOMAINS", False)
    SECURE_HSTS_PRELOAD: bool = _env_bool("DJANGO_HSTS_PRELOAD", False)
    CSRF_COOKIE_SECURE: bool = SECURE_SSL_REDIRECT

# The rate limit and the scanner ban. Django's default of 300 entries would let a few hundred
# addresses push the bans out within a minute.
CACHES: dict = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "fare",
        "OPTIONS": {"MAX_ENTRIES": 10_000},
    }
}

FARE_ARTIFACTS_DIR: Path = Path(os.getenv("FARE_ARTIFACTS_DIR", str(REPO_DIR / "artifacts")))
FARE_RATE_LIMIT: int = int(os.getenv("FARE_RATE_LIMIT", "30"))
RATELIMIT_TRUSTED_FORWARDED_HOPS: int = int(os.getenv("RATELIMIT_TRUSTED_FORWARDED_HOPS", "0"))
RATELIMIT_TRUST_CF_HEADER: bool = _env_bool("RATELIMIT_TRUST_CF_HEADER", False)
PROBE_BAN_SECONDS: int = int(os.getenv("PROBE_BAN_SECONDS", "3600"))

LOG_FORMAT: str = os.getenv("LOG_FORMAT", "plain" if DEBUG else "json")
if LOG_FORMAT not in {"json", "plain"}:
    LOG_FORMAT = "plain" if DEBUG else "json"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOGGING: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.json.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
        "plain": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
    },
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": LOG_FORMAT}},
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {"django": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False}},
}
