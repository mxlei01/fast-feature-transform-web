"""The Cloudflare Tunnel's routing for this site, and a command that pushes it to Cloudflare.

The tunnel was created in Cloudflare's dashboard, so Cloudflare holds its routing and the connector in
the cluster needs only the tunnel's token. This file is where that routing is written down: change it
here, then push it with ``make tunnel-routing``. Rules match top to bottom. The two hostname-less rules
come first, so every public hostname, present or future, answers ``/healthz`` (uncached work meant for
the cluster's probes) and ``/admin`` (which the site does not serve) with a 404 at Cloudflare's edge,
before the request reaches the cluster.

    CF_ACCOUNT_ID=... CF_TUNNEL_ID=... FARE_HOST=... FARE_STAGING_HOST=... make tunnel-routing

``make tunnel-routing ARGS=--dry-run`` prints the routing without pushing it.

Environment:
    CF_ACCOUNT_ID: The Cloudflare account's ID.
    CF_TUNNEL_ID: The tunnel's ID (``cloudflared tunnel list``).
    FARE_HOST, FARE_STAGING_HOST: The public hostnames of prod and staging.
    CF_TOKEN_FILE: A file holding an API token with Account, Cloudflare Tunnel, Edit (default
        ``.cf-token-edit`` at the repository root, which git ignores). The token is never printed.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_DIR: Path = Path(__file__).resolve().parents[2]
API: str = "https://api.cloudflare.com/client/v4"
HOSTNAME = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
HEX_ID = re.compile(r"^[0-9a-f]{32}$")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def ingress(prod_host: str, staging_host: str) -> list[dict[str, str]]:
    """Return the tunnel's ingress rules, in the order Cloudflare matches them.

    Args:
        prod_host: Prod's public hostname.
        staging_host: Staging's public hostname.

    Returns:
        The rules: the edge blocks, one rule per hostname, then the required catch-all.

    """
    return [
        {"path": "^/healthz", "service": "http_status:404"},
        {"path": "^/admin", "service": "http_status:404"},
        # The connector runs in the prod namespace; staging's Service is reached across namespaces.
        {"hostname": prod_host, "service": "http://fare-web.fare-web.svc.cluster.local:80"},
        {"hostname": staging_host, "service": "http://fare-web.fare-web-staging.svc.cluster.local:80"},
        {"service": "http_status:404"},
    ]


def _require(name: str, pattern: re.Pattern[str]) -> str:
    """Read an environment variable that must match a pattern.

    Args:
        name: The variable.
        pattern: What its value must look like.

    Returns:
        The value, lowercased.

    """
    value: str = os.environ.get(name, "").strip().lower()
    if not pattern.match(value):
        sys.exit(f"{name} is missing or malformed")
    return value


def main() -> int:
    """Build the routing from the environment, print it, and push it unless ``--dry-run`` is given.

    Returns:
        The exit status.

    """
    prod_host: str = _require("FARE_HOST", HOSTNAME)
    staging_host: str = _require("FARE_STAGING_HOST", HOSTNAME)
    if prod_host == staging_host:
        sys.exit("FARE_HOST and FARE_STAGING_HOST must differ")
    body: dict[str, Any] = {"config": {"ingress": ingress(prod_host, staging_host)}}
    print(json.dumps(body, indent=2))
    if "--dry-run" in sys.argv[1:]:
        return 0
    account: str = _require("CF_ACCOUNT_ID", HEX_ID)
    tunnel: str = _require("CF_TUNNEL_ID", UUID)
    token_file = Path(os.environ.get("CF_TOKEN_FILE", REPO_DIR / ".cf-token-edit"))
    token: str = token_file.read_text().strip()
    request = urllib.request.Request(  # noqa: S310 - a fixed https URL
        f"{API}/accounts/{account}/cfd_tunnel/{tunnel}/configurations",
        data=json.dumps(body).encode(),
        method="PUT",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310
            reply: dict[str, Any] = json.load(response)
    except urllib.error.HTTPError as error:
        reply = json.load(error)
    if not reply.get("success"):
        print(json.dumps(reply.get("errors", []), indent=2), file=sys.stderr)
        return 1
    print(f"pushed: the tunnel's routing is now version {reply['result'].get('version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
