"""The smoke check waits out refused connections, and only those, before it judges the site."""

import socket
import types
import urllib.error

import pytest

from deploy.manifests.base import smoke_check


def test_the_wait_retries_refused_connections_until_the_site_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fetch(path: str) -> tuple[int, bytes]:
        asked.append(path)
        if len(asked) < 3:
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        return 503, b""

    monkeypatch.setattr(smoke_check, "fetch", fetch)
    monkeypatch.setattr(smoke_check.time, "sleep", lambda seconds: None)
    smoke_check.wait_for_site()
    assert asked == ["/healthz"] * 3  # any answer, even a 503, ends the wait; the checks judge it


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),  # no answer in time, once connected
        urllib.error.URLError(TimeoutError()),  # no answer to the connection itself
        urllib.error.URLError(socket.gaierror(-2, "Name or service not known")),
        urllib.error.URLError("unknown url type: htp"),
    ],
)
def test_the_wait_ends_at_any_other_error(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    asked: list[str] = []

    def fetch(path: str) -> tuple[int, bytes]:
        asked.append(path)
        raise error

    monkeypatch.setattr(smoke_check, "fetch", fetch)
    monkeypatch.setattr(smoke_check.time, "sleep", lambda seconds: pytest.fail("retried"))
    smoke_check.wait_for_site()
    assert asked == ["/healthz"]


def test_the_wait_gives_up_after_smoke_wait_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    now: list[float] = [0.0]

    def fetch(path: str) -> tuple[int, bytes]:
        asked.append(path)
        assert len(asked) <= 100, "the wait never ended"  # an AssertionError gets past the wait's handlers
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    def sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(smoke_check, "fetch", fetch)
    monkeypatch.setattr(smoke_check, "WAIT", 30)
    monkeypatch.setattr(smoke_check, "time", types.SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    smoke_check.wait_for_site()
    assert asked == ["/healthz"] * 15  # an attempt every 2 s, from 0 s to 28 s


def test_the_checks_run_after_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(smoke_check, "BASE_URL", "http://fare-web")
    monkeypatch.setattr(smoke_check, "wait_for_site", lambda: calls.append("wait"))
    monkeypatch.setattr(smoke_check, "CHECKS", [("health", lambda: calls.append("check"))])
    assert smoke_check.main() == 0
    assert calls == ["wait", "check"]
