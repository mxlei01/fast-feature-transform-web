"""The scanner honeypot: a scanner's path gets a 404 and its sender a short ban, while visitors are never banned."""

import pytest
from django.test import Client
from pytest_django.fixtures import Settings

from fare import probe_guard
from fare.probe_guard import is_scanner_path

# Public addresses: the documentation ranges are not globally routable, so they would never be banned.
SCANNER: str = "8.8.8.8"
VISITOR: str = "9.9.9.9"
CONNECTOR: str = "10.42.0.7"


@pytest.mark.parametrize(
    "path",
    ["/wp-login.php", "/WP-ADMIN/setup-config.php", "/.env", "/app/.git/config", "/cgi-bin/luci", "/config.json"],
)
def test_scanner_paths_are_recognized_in_any_case(path: str) -> None:
    assert is_scanner_path(path)


@pytest.mark.parametrize(
    "path", ["/", "/healthz", "/api/predict", "/api/pipeline", "/static/fare/site.css", "/environment"]
)
def test_the_sites_own_paths_are_not_scanner_paths(path: str) -> None:
    assert not is_scanner_path(path)


def test_a_scanner_gets_a_404_and_then_nothing_else(client: Client) -> None:
    assert client.get("/", REMOTE_ADDR=SCANNER).status_code == 200
    probe = client.get("/wp-login.php", REMOTE_ADDR=SCANNER)
    assert probe.status_code == 404
    assert "Content-Security-Policy" in probe  # answered inside the security headers
    assert client.get("/", REMOTE_ADDR=SCANNER).status_code == 404
    assert client.get("/", REMOTE_ADDR=VISITOR).status_code == 200


def test_a_request_a_browser_sent_from_another_site_bans_no_one(client: Client) -> None:
    probe = client.get("/wp-login.php", REMOTE_ADDR=VISITOR, HTTP_SEC_FETCH_SITE="cross-site")
    assert probe.status_code == 404
    assert client.get("/", REMOTE_ADDR=VISITOR).status_code == 200


def test_a_private_address_is_never_banned(client: Client) -> None:
    # Behind the tunnel, with Cloudflare's header not trusted, every visitor arrives from the connector.
    assert client.get("/.env", REMOTE_ADDR=CONNECTOR).status_code == 404
    assert client.get("/", REMOTE_ADDR=CONNECTOR).status_code == 200


def test_behind_cloudflare_the_ban_follows_the_visitor_not_the_connector(client: Client, settings: Settings) -> None:
    settings.RATELIMIT_TRUST_CF_HEADER = True
    assert client.get("/.git/config", REMOTE_ADDR=CONNECTOR, HTTP_CF_CONNECTING_IP=SCANNER).status_code == 404
    assert client.get("/", REMOTE_ADDR=CONNECTOR, HTTP_CF_CONNECTING_IP=SCANNER).status_code == 404
    assert client.get("/", REMOTE_ADDR=CONNECTOR, HTTP_CF_CONNECTING_IP=VISITOR).status_code == 200


def test_the_ban_lasts_probe_ban_seconds(client: Client, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.PROBE_BAN_SECONDS = 120
    calls: list[tuple[str, object, int]] = []
    real_set = probe_guard.cache.set
    monkeypatch.setattr(
        probe_guard.cache,
        "set",
        lambda key, value, timeout: calls.append((key, value, timeout)) or real_set(key, value, timeout),
    )
    client.get("/phpmyadmin/", REMOTE_ADDR=SCANNER)
    assert calls == [(f"probe-ban:{SCANNER}", True, 120)]
