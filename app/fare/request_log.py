"""One log record per request: method, path, status and duration, plus who asked.

The message itself reads ``GET / 200 12ms`` so plain development logs are useful; the same values,
with the client's address and user agent, ride along as fields for the JSON format. The path and the
headers come from the client, so each is cut to ``MAX_FIELD_LEN`` characters and its control
characters are escaped, so a request cannot start a forged line in the plain format. The health
check is skipped.
"""

import logging
import time
from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

from fare.ratelimit import client_ip

logger = logging.getLogger("fare.request")

SKIP_PATHS: frozenset[str] = frozenset({"/healthz"})
MAX_FIELD_LEN: int = 200


def _clean(value: str) -> str:
    """Cut a client-supplied value and escape everything outside printable ASCII.

    Args:
        value: The value.

    Returns:
        At most ``MAX_FIELD_LEN`` characters of it, with newlines, tabs and the like written as escapes.

    """
    return value[:MAX_FIELD_LEN].encode("unicode_escape").decode("ascii")


class RequestLogMiddleware:
    """Log every request once it has a response."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        """Wrap the rest of the stack.

        Args:
            get_response: The next layer.

        """
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Time the request and log it.

        Args:
            request: The request.

        Returns:
            The response, unchanged.

        """
        if request.path in SKIP_PATHS:
            return self.get_response(request)
        start: float = time.monotonic()
        response: HttpResponse = self.get_response(request)
        duration_ms: int = int((time.monotonic() - start) * 1000)
        method: str = _clean(request.method or "")
        path: str = _clean(request.path)
        logger.info(
            "%s %s %d %dms",
            method,
            path,
            response.status_code,
            duration_ms,
            extra={
                "method": method,
                "path": path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "ip": _clean(client_ip(request)),
                "user_agent": _clean(request.META.get("HTTP_USER_AGENT", "")),
            },
        )
        return response
