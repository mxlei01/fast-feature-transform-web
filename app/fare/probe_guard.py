"""Scanner honeypot: requests for paths this site can never serve get a 404, and their sender a short ban.

The site is one small Django app. It runs no PHP, WordPress or phpMyAdmin and serves no ``.env``,
``.git`` or credential files, so a request for one of ``SCANNER_PATHS`` is a scanner fingerprinting some
other stack. It gets a bare 404, and its sender is banned for ``PROBE_BAN_SECONDS``: every later request
from that address (for IPv6, its whole /64) gets the same 404 before the site does any work. The site's
static files are the exception, since WhiteNoise serves them before this guard runs.

The ban lives in this process's cache, so it only sheds an active scan. It is short on purpose: one
public address can be many people behind a carrier's or an office's NAT. Lasting blocks belong at
Cloudflare's edge. Two kinds of request get the 404 but never a ban:

- A request a browser sent. A page that embeds ``<img src="https://.../wp-login.php">`` makes each
  visitor's browser ask for it, and so does a link opened from mail or chat, or an extension that
  checks every site for ``/.git``. Current browsers label every request with ``Sec-Fetch-Site``;
  scanners send no ``Sec-Fetch`` headers, and one that forges the header still gets the 404. Browsers
  older than Safari 16.4, crawlers and link previewers send no such header, so a hostile page or a
  planted link can still get them banned.
- A request from a private address, such as the tunnel connector's when the site does not trust
  Cloudflare's visitor header. Banning that address would refuse every visitor.

The 404s are marked ``private, no-store``: a ban makes the answer depend on who asked, so no shared
cache, Cloudflare's included, may keep one.
"""

import ipaddress
from collections.abc import Callable

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, HttpResponseNotFound

from fare.ratelimit import client_ip, client_key

# Matched anywhere in the lowercased path. Only fragments no URL on this site can contain belong here,
# so a mistyped real URL never bans a visitor. Never add a bare "/.": real clients fetch /.well-known/.
SCANNER_PATHS: tuple[str, ...] = (
    ".php",
    "/wp-admin",
    "/wp-login",
    "/wp-content",
    "/wp-includes",
    "/phpmyadmin",
    "/phpunit",
    "/.env",
    "/.git",
    "/.svn/",
    "/.ht",
    "/.ds_store",
    "/.aws/",
    "/.ssh/",
    "/cgi-bin/",
    "/actuator/",
    "/solr/",
    "/boaform/",
    "/config.json",
)


def is_scanner_path(path: str) -> bool:
    """Tell whether a path is one only a scanner would ask this site for.

    Args:
        path: The request's path.

    Returns:
        True if it holds one of ``SCANNER_PATHS``, ignoring case.

    """
    lowered: str = path.lower()
    return any(fragment in lowered for fragment in SCANNER_PATHS)


def _ban_key(address: str) -> str:
    """Return the cache key that marks an address, or an IPv6 address's /64, as banned.

    Args:
        address: The client's address.

    Returns:
        The key.

    """
    return f"probe-ban:{client_key(address)}"


def _bannable(address: str) -> bool:
    """Tell whether an address may be banned: a known, globally routable one.

    Args:
        address: The client's address, or ``"?"``.

    Returns:
        True for a public address.

    """
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def _refusal() -> HttpResponse:
    """Return the guard's bare 404, which no shared cache may keep.

    Returns:
        The 404.

    """
    response: HttpResponse = HttpResponseNotFound()
    response["Cache-Control"] = "private, no-store"
    return response


class ProbeGuardMiddleware:
    """Refuse scanner paths, and every request from an address that recently asked for one."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        """Wrap the rest of the stack.

        Args:
            get_response: The next layer.

        """
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Answer a banned address or a scanner path with a 404; pass everything else on.

        Args:
            request: The request.

        Returns:
            A bare 404, or the rest of the stack's response.

        """
        address: str = client_ip(request)
        if cache.get(_ban_key(address)):
            return _refusal()
        if not is_scanner_path(request.path):
            return self.get_response(request)
        if _bannable(address) and not request.META.get("HTTP_SEC_FETCH_SITE"):
            cache.set(_ban_key(address), True, settings.PROBE_BAN_SECONDS)
        return _refusal()
