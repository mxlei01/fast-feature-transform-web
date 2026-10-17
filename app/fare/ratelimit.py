"""Who the client is, and a per-client fixed-window limit on how often it may ask.

``client_ip`` is the one place the client's address is decided, so the request log, the limiter and
the scanner ban always agree. Behind a Cloudflare Tunnel every request reaches the pod from the
connector, so with ``RATELIMIT_TRUST_CF_HEADER`` on, the visitor's address comes from
``CF-Connecting-IP``, which Cloudflare's edge sets on every request and overwrites if the visitor sent
one. Otherwise X-Forwarded-For, a list any client can prepend to, is believed only for the entries
appended by the proxies we run (``RATELIMIT_TRUSTED_FORWARDED_HOPS``, counted from the right); with no
trusted proxy it is ignored and the socket's address is used.

The budget and the ban are kept per ``client_key``: an IPv4 address, or an IPv6 address's /64.
"""

import ipaddress

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest

WINDOW_SECONDS: int = 60
IPV6_CLIENT_PREFIX: int = 64


def _parse_ip(value: str) -> str | None:
    """Return an address in its one canonical spelling, or None for anything else.

    An IPv6 zone (the text after ``%``, as in ``fe80::1%eth0``) is refused: it can be any text, and no
    visitor's address carries one.

    Args:
        value: The candidate.

    Returns:
        The address as ``ipaddress`` writes it, or None.

    """
    try:
        address: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(value)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.scope_id is not None:
        return None
    return str(address)


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
        edge: str | None = _parse_ip(request.META.get("HTTP_CF_CONNECTING_IP", "").strip())
        if edge:
            return edge
    hops: int = getattr(settings, "RATELIMIT_TRUSTED_FORWARDED_HOPS", 0)
    if hops > 0:
        entries: list[str] = [
            part.strip() for part in request.META.get("HTTP_X_FORWARDED_FOR", "").split(",") if part.strip()
        ]
        if entries:
            candidate: str | None = _parse_ip(entries[max(len(entries) - hops, 0)])
            if candidate:
                return candidate
    return request.META.get("REMOTE_ADDR") or "?"


def client_key(address: str) -> str:
    """Return what a client's budget and ban are kept under: an IPv4 address, or an IPv6 address's /64.

    An IPv6 home or server holds a whole /64 and can send each request from a new address in it, so
    counting per address would set no limit. A /64 is one household or one machine, as one IPv4 address is.

    Args:
        address: The address ``client_ip`` returned.

    Returns:
        The key. Anything that is not an address, such as ``"?"``, comes back unchanged.

    """
    try:
        parsed: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(address)
    except ValueError:
        return address
    if isinstance(parsed, ipaddress.IPv6Address):
        if parsed.ipv4_mapped is not None:  # an IPv4 client written as IPv6; its /64 would hold all of them
            return str(parsed.ipv4_mapped)
        return str(ipaddress.IPv6Network((parsed, IPV6_CLIENT_PREFIX), strict=False))
    return str(parsed)


def limited(request: HttpRequest, bucket: str, limit: int) -> bool:
    """Count a request against its client's budget for this minute.

    Args:
        request: The request.
        bucket: Name of the budget, so different endpoints can count separately.
        limit: Requests allowed per client per minute.

    Returns:
        True when this request goes over the limit and should be refused.

    """
    key: str = f"ratelimit:{bucket}:{client_key(client_ip(request))}"
    cache.add(key, 0, WINDOW_SECONDS)
    try:
        count: int = cache.incr(key)
    except ValueError:  # the window expired between add and incr
        cache.set(key, 1, WINDOW_SECONDS)
        count = 1
    return count > limit
