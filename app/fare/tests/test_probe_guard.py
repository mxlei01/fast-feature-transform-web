"""The scanner honeypot: a scanner's path gets a 404 and its sender a short ban; no browser or private address does."""

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
    [
        "/wp-login.php",
        "/WP-ADMIN/setup-config.php",
        "/index.php",
        "/phpinfo.php",
        "/pma/index.php",
        "/.env",
        "/app/.git/config",
        "/.git",
        "/.gitignore",
        "/.svn/entries",
        "/.htaccess",
        "/.DS_Store",
        "/cgi-bin/luci",
        "/config.json",
    ],
)
def test_scanner_paths_are_recognized_in_any_case(path: str) -> None:
    assert is_scanner_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/healthz",
        "/api/predict",
        "/api/pipeline",
        "/static/fare/app.css",
        "/environment",
        "/.well-known/security.txt",
        "/.well-known/change-password",
    ],
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


@pytest.mark.parametrize("site", ["cross-site", "same-site", "same-origin", "none"])
def test_a_request_a_browser_sent_bans_no_one(client: Client, site: str) -> None:
    # cross-site: a hostile page's <img>; none: a link opened from mail or chat, or an extension's check.
    probe = client.get("/wp-login.php", REMOTE_ADDR=VISITOR, HTTP_SEC_FETCH_SITE=site)
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


def test_an_ipv6_ban_covers_the_senders_slash_64(client: Client, settings: Settings) -> None:
    # One IPv6 host can use every address in its /64, so a ban on one address alone would not hold.
    settings.RATELIMIT_TRUST_CF_HEADER = True
    assert client.get("/.env", REMOTE_ADDR=CONNECTOR, HTTP_CF_CONNECTING_IP="2a01:4f8:1:2::99").status_code == 404
    assert client.get("/", REMOTE_ADDR=CONNECTOR, HTTP_CF_CONNECTING_IP="2a01:4f8:1:2::9a").status_code == 404
    assert client.get("/", REMOTE_ADDR=CONNECTOR, HTTP_CF_CONNECTING_IP="2a01:4f8:1:3::9a").status_code == 200


def test_a_banned_address_still_gets_the_stylesheet(client: Client) -> None:
    # The guard's 404s are no-store, but under a cache rule that ignores that header, Cloudflare would keep a
    # 404 here by the .css extension under the URL alone and serve it to every visitor.
    client.get("/wp-login.php", REMOTE_ADDR=SCANNER)
    stylesheet = client.get("/static/fare/app.css", REMOTE_ADDR=SCANNER)
    stylesheet.close()  # WhiteNoise hands back an open file
    assert stylesheet.status_code == 200
    assert client.get("/", REMOTE_ADDR=SCANNER).status_code == 404


def test_the_guards_404s_are_never_cached(client: Client) -> None:
    probe = client.get("/wp-content/themes/style.css", REMOTE_ADDR=SCANNER)
    banned = client.get("/", REMOTE_ADDR=SCANNER)
    assert probe.status_code == banned.status_code == 404
    assert probe["Cache-Control"] == banned["Cache-Control"] == "private, no-store"


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


def test_a_ban_outlasts_a_few_hundred_other_addresses(client: Client) -> None:
    # Django's cache keeps 300 entries by default, and makes room by dropping the least recently used.
    client.get("/.env", REMOTE_ADDR=SCANNER)
    for number in range(400):
        client.get("/.env", REMOTE_ADDR=f"8.8.{number // 200}.{number % 200 + 1}")
    assert client.get("/", REMOTE_ADDR=SCANNER).status_code == 404
