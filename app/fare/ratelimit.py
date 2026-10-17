"""Who the client is, and a per-client fixed-window limit on how often it may ask.

``client_ip`` is the one place the client's address is decided, so the request log, the limiter and
the scanner ban always agree. Behind a Cloudflare Tunnel every request reaches the pod from the
connector, so with ``RATELIMIT_TRUST_CF_HEADER`` on, the visitor's address comes from
``CF-Connecting-IP``, which Cloudflare's edge sets on every request and overwrites if the visitor sent
one. Otherwise X-Forwarded-For, a list any client can prepend to, is believed only for the entries
appended by the proxies we run (``RATELIMIT_TRUSTED_FORWARDED_HOPS``, counted from the right); with no
trusted proxy it is ignored and the socket's address is used.
"""

import ipaddress

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest

WINDOW_SECONDS: int = 60


def _valid_ip(value: str) -> bool:
    """Tell whether a string is an IPv4 or IPv6 address.

    Args:
        value: The candidate.

    Returns:
        True if it parses as an address.

    """
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def client_ip(request: HttpRequest) -> str:
    """Return the client's address: Cloudflare's record of it when trusted, else our own proxies' entries.

    Turn ``RATELIMIT_TRUST_CF_HEADER`` on only where every way to the site passes through Cloudflare or
    strips the header (deploy/README.md), since a client that reaches the pod directly could set it.

    Args:
        request: The request.

    Returns:
        The address, or ``"?"`` when none is known.

    """
    if getattr(settings, "RATELIMIT_TRUST_CF_HEADER", False):
        edge: str = request.META.get("HTTP_CF_CONNECTING_IP", "").strip()
        if edge and _valid_ip(edge):
            return edge
    hops: int = getattr(settings, "RATELIMIT_TRUSTED_FORWARDED_HOPS", 0)
    if hops > 0:
        entries: list[str] = [
            part.strip() for part in request.META.get("HTTP_X_FORWARDED_FOR", "").split(",") if part.strip()
        ]
        if entries:
            candidate: str = entries[max(len(entries) - hops, 0)]
            if _valid_ip(candidate):
                return candidate
    return request.META.get("REMOTE_ADDR") or "?"


def limited(request: HttpRequest, bucket: str, limit: int) -> bool:
    """Count a request against its client's budget for this minute.

    Args:
        request: The request.
        bucket: Name of the budget, so different endpoints can count separately.
        limit: Requests allowed per client per minute.

    Returns:
        True when this request goes over the limit and should be refused.

    """
    key: str = f"ratelimit:{bucket}:{client_ip(request)}"
    cache.add(key, 0, WINDOW_SECONDS)
    try:
        count: int = cache.incr(key)
    except ValueError:  # the window expired between add and incr
        cache.set(key, 1, WINDOW_SECONDS)
        count = 1
    return count > limit
