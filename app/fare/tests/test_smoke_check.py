"""The smoke check waits out refused connections, and only those, before it judges the site."""

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


def test_the_wait_ends_at_any_other_error(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fetch(path: str) -> tuple[int, bytes]:
        asked.append(path)
        raise TimeoutError

    monkeypatch.setattr(smoke_check, "fetch", fetch)
    smoke_check.wait_for_site()
    assert asked == ["/healthz"]
