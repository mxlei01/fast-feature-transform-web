"""WSGI entry point: ``gunicorn app.wsgi:application`` from inside ``app/``."""

import os
import sys
from pathlib import Path

from django.core.wsgi import get_wsgi_application

# The served model's feature code lives in the training package beside app/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")

application = get_wsgi_application()
