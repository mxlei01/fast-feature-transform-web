"""Settings for the test suite: the production settings, with a throwaway key and no HTTPS redirect."""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "tests-only-not-a-secret")
os.environ.setdefault("DJANGO_SSL_REDIRECT", "false")

from app.settings import *  # noqa: E402, F403
