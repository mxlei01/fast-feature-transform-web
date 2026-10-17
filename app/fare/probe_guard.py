"""Scanner honeypot: requests for paths this site can never serve get a 404, and their sender a short ban.

The site is one small Django app. It runs no WordPress, PHP or phpMyAdmin and never serves dotfiles or
cloud credentials, so a request for any of those is a scanner fingerprinting some other stack. It gets a
bare 404, and its sender's address is banned for ``PROBE_BAN_SECONDS``: every later request from that
address gets the same 404 before the site does any work.

The ban lives in this process's cache, so it only sheds an active scan. It is short on purpose: one
public address can be many people behind a carrier's or an office's NAT. Lasting blocks belong at
Cloudflare's edge. Two kinds of request get the 404 but never a ban:

- A request a browser sent from another site. A page that embeds ``<img src="https://.../wp-login.php">``
  makes each visitor's browser ask for it, which would get those visitors banned. Browsers label such
  requests ``Sec-Fetch-Site: cross-site``; scanners send no ``Sec-Fetch`` headers.
- A request from a private address, such as the tunnel connector's when the site does not trust
  Cloudflare's visitor header. Banning that address would refuse every visitor.
"""

import ipaddress
from collections.abc import Callable

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, HttpResponseNotFound

from fare.ratelimit import client_ip

# Matched anywhere in the lowercased path. Only paths that cannot exist on this site belong here, so a
# mistyped real URL never bans a visitor.
SCANNER_PATHS: tuple[str, ...] = (
    "/wp-admin",
    "/wp-login",
    "/wp-content",
    "/wp-includes",
    "/xmlrpc.php",
    "/.env",
    "/.git/",
    "/.aws/",
    "/.ssh/",
    "/phpmyadmin",
    "/phpunit",
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
    """Return the cache key that marks an address as banned.

    Args:
        address: The client's address.

    Returns:
        The key.

    """
    return f"probe-ban:{address}"


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
            return HttpResponseNotFound()
        if not is_scanner_path(request.path):
            return self.get_response(request)
        if _bannable(address) and request.META.get("HTTP_SEC_FETCH_SITE") != "cross-site":
            cache.set(_ban_key(address), True, settings.PROBE_BAN_SECONDS)
        return HttpResponseNotFound()
