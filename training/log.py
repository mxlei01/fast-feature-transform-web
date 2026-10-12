"""Logging for the command line, in the web app's two formats.

``LOG_FORMAT=json`` writes one JSON object per line through python-json-logger, as the web app does
in production; anything else writes plain lines. ``LOG_LEVEL`` sets the level (INFO by default).
"""

import logging.config
import os

FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure() -> None:
    """Send every logger's records to stderr in the chosen format."""
    style: str = "json" if os.getenv("LOG_FORMAT", "plain") == "json" else "plain"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": "pythonjsonlogger.json.JsonFormatter", "format": FORMAT},
                "plain": {"format": FORMAT},
            },
            "handlers": {"console": {"class": "logging.StreamHandler", "formatter": style}},
            "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
        }
    )
