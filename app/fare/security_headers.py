"""Browser hardening headers for every response.

The pages need no JavaScript at all, so the Content Security Policy allows none: a script injected
through any field could not run. Styles and images come only from this site, forms may only post
back to it, and no other site may frame it. ``setdefault`` lets a view send its own policy if one
ever needs to.

Error responses are marked ``no-store`` unless they set their own Cache-Control. Cloudflare keeps a
404 with no Cache-Control for three minutes under a URL ending in ``.css``, and with its default URL
normalization it may file ``/static/fare/./app.css``, which reaches the site unchanged and gets a 404,
under the stylesheet's own URL.
"""

from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

CSP: str = (
    "default-src 'none'; style-src 'self'; img-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)
PERMISSIONS_POLICY: str = "camera=(), microphone=(), geolocation=(), payment=(), usb=()"


class SecurityHeadersMiddleware:
    """Add the Content Security Policy and Permissions Policy, and keep errors out of shared caches."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        """Wrap the rest of the stack.

        Args:
            get_response: The next layer.

        """
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Add the headers to the response.

        Args:
            request: The request.

        Returns:
            The response with the headers set.

        """
        response: HttpResponse = self.get_response(request)
        response.setdefault("Content-Security-Policy", CSP)
        response.setdefault("Permissions-Policy", PERMISSIONS_POLICY)
        if response.status_code >= 400:
            response.setdefault("Cache-Control", "no-store")
        return response
