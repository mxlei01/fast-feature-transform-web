"""Smoke checks for a deployed fare site, using only the standard library.

ArgoCD runs this as a PostSync Job on staging against the Service in the same namespace, and a
non-zero exit marks the sync failed. It needs nothing beyond the site's own image. It sends
``Host: localhost``, so run it by hand against gunicorn itself, for example through
``kubectl port-forward``:

    FARE_BASE_URL=http://host:port python deploy/manifests/base/smoke_check.py

Environment:
    FARE_BASE_URL: The site's base URL (required).
    SMOKE_TIMEOUT: Seconds allowed per request (default 30).
    SMOKE_WAIT: Seconds to wait for the site to accept a connection before checking (default 30).
"""

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

BASE_URL: str = os.environ.get("FARE_BASE_URL", "").rstrip("/")
TIMEOUT: int = int(os.environ.get("SMOKE_TIMEOUT", "30"))
WAIT: int = int(os.environ.get("SMOKE_WAIT", "30"))

# A trip the committed model knows: Midtown Center to Upper East Side North on a weekday morning.
TRIP: dict[str, Any] = {
    "pickup_zone": 161,
    "dropoff_zone": 236,
    "pickup_time": "2026-08-12T09:15",
    "trip_distance": 2.4,
}
STAGES: list[str] = ["categories", "indices", "distance", "features"]


def fetch(path: str, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
    """Request a path the way the proxy in front of the site presents it.

    ``Host: localhost`` is always allowed, and ``X-Forwarded-Proto: https`` keeps the HTTPS
    redirect from answering in place of the view.

    Args:
        path: The path, starting with ``/``.
        body: A JSON body to POST, or None to GET.

    Returns:
        The status and the response body.

    """
    headers: dict[str, str] = {"Host": "localhost", "X-Forwarded-Proto": "https"}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers)  # noqa: S310 - URL from the environment
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310  # nosec B310
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_for_site() -> None:
    """Wait up to ``SMOKE_WAIT`` seconds for the site to accept a connection.

    The cluster adds a new pod's address to the site's NetworkPolicy a few seconds after the pod starts
    and refuses its connections until then, so the Job's first requests can be refused although the site
    is fine. Only a refused connection is retried; any answer, or any other error, ends the wait, and the
    checks then judge it.
    """
    deadline: float = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        try:
            fetch("/healthz")
        except urllib.error.URLError:
            time.sleep(2)
        except ValueError, OSError:
            return
        else:
            return


def check_health() -> None:
    """Check that the model is loaded."""
    status, body = fetch("/healthz")
    assert status == 200, f"/healthz answered {status}: {body[:200]!r}"
    assert json.loads(body) == {"status": "ok"}, f"/healthz answered {body[:200]!r}"


def check_page() -> None:
    """Check that the page renders its form."""
    status, body = fetch("/")
    assert status == 200, f"/ answered {status}"
    assert b"<form" in body, "/ has no form"


def check_prediction() -> None:
    """Check that a trip is priced, with every stage of the pipeline traced."""
    status, body = fetch("/api/predict", TRIP)
    assert status == 200, f"/api/predict answered {status}: {body[:200]!r}"
    answer: dict[str, Any] = json.loads(body)
    fare: float = answer["fare"]
    assert math.isfinite(fare) and fare > 0, f"fare {fare!r}"
    assert [step["stage"] for step in answer["trace"]] == STAGES, f"trace {answer['trace']!r}"


def check_pipeline() -> None:
    """Check that the pipeline description lists every stage."""
    status, body = fetch("/api/pipeline")
    assert status == 200, f"/api/pipeline answered {status}"
    assert [stage["stage"] for stage in json.loads(body)["stages"]] == STAGES


CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("health", check_health),
    ("page", check_page),
    ("prediction", check_prediction),
    ("pipeline", check_pipeline),
]


def main() -> int:
    """Run every check and report each one.

    Returns:
        0 when all pass, 1 otherwise.

    """
    if not BASE_URL:
        print("set FARE_BASE_URL", file=sys.stderr)
        return 1
    wait_for_site()
    failed: int = 0
    for name, check in CHECKS:
        try:
            check()
        except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
            failed += 1
            print(f"FAIL {name}: {error}", flush=True)
        else:
            print(f"ok   {name}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
