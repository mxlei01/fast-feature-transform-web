"""Browser hardening headers for every response.

The pages need no JavaScript at all, so the Content Security Policy allows none: a script injected
through any field could not run. Styles and images come only from this site, forms may only post
back to it, and no other site may frame it. ``setdefault`` lets a view send its own policy if one
ever needs to.
"""

from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

CSP: str = (
    "default-src 'none'; style-src 'self'; img-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)
PERMISSIONS_POLICY: str = "camera=(), microphone=(), geolocation=(), payment=(), usb=()"


class SecurityHeadersMiddleware:
    """Add the Content Security Policy and Permissions Policy."""

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
        return response
