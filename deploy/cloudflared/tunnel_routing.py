"""The Cloudflare Tunnel's routing for this site, and a command that pushes it to Cloudflare.

The tunnel was created in Cloudflare's dashboard, so Cloudflare stores its routing and sends it to the
connector in the cluster, which needs only the tunnel's token. This file is where that routing is written
down: change it here, check it with ``make tunnel-routing``, which only prints it, then push it with
``make tunnel-routing ARGS=--push``. A push replaces the tunnel's whole configuration, including anything
set in the dashboard.

The connector matches each request against the rules top to bottom. The two hostname-less rules come
first, so for every public hostname, present or future, the connector itself answers ``/healthz``
(uncached work meant for the cluster's probes) and ``/admin`` (which the site does not serve) with a 404.
Such a request still crosses the tunnel but never reaches the site's pods; stopping it at Cloudflare's
edge takes a WAF rule on the zone.

    CF_ACCOUNT_ID=... CF_TUNNEL_ID=... FARE_HOST=... FARE_STAGING_HOST=... make tunnel-routing ARGS=--push

Only ``--push`` pushes, so a mistyped flag or make variable cannot.

Environment:
    CF_ACCOUNT_ID: The Cloudflare account's ID.
    CF_TUNNEL_ID: The tunnel's ID, shown on its page in Cloudflare's dashboard.
    FARE_HOST, FARE_STAGING_HOST: The public hostnames of prod and staging.
    CF_TOKEN_FILE: A file holding only an API token with Account, Cloudflare Tunnel, Edit (default
        ``~/.config/cloudflare/edit-token``). Keep it outside the repository: git ignores ``.cf-token*``,
        but a synced folder would still copy it. The token is never printed.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOKEN_FILE: Path = Path.home() / ".config" / "cloudflare" / "edit-token"
API: str = "https://api.cloudflare.com/client/v4"
HOSTNAME = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
HEX_ID = re.compile(r"^[0-9a-f]{32}$")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# RFC 6750's b64token, which every bearer token matches. Anything else in the token file would reach the
# Authorization header, and http.client's refusal of it quotes the whole header, token included.
BEARER_TOKEN = re.compile(r"[A-Za-z0-9._~+/-]+=*")


def ingress(prod_host: str, staging_host: str) -> list[dict[str, str]]:
    """Return the tunnel's ingress rules, in the order the connector matches them.

    Args:
        prod_host: Prod's public hostname.
        staging_host: Staging's public hostname.

    Returns:
        The rules: the two path blocks, one rule per hostname, then the required catch-all.

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


def main(argv: list[str] | None = None) -> int:
    """Build the routing from the environment, print it, and push it only when ``--push`` is given.

    Args:
        argv: The arguments, without the program's name; the process's own when None.

    Returns:
        The exit status.

    """
    # An unknown or mistyped flag exits here, before anything is read or pushed.
    parser = argparse.ArgumentParser(description="Print the tunnel's routing; --push sends it to Cloudflare.")
    parser.add_argument("--push", action="store_true", help="replace the tunnel's live routing with this one")
    push: bool = parser.parse_args(argv).push
    prod_host: str = _require("FARE_HOST", HOSTNAME)
    staging_host: str = _require("FARE_STAGING_HOST", HOSTNAME)
    if prod_host == staging_host:
        sys.exit("FARE_HOST and FARE_STAGING_HOST must differ")
    body: dict[str, Any] = {"config": {"ingress": ingress(prod_host, staging_host)}}
    print(json.dumps(body, indent=2))
    if not push:
        return 0
    account: str = _require("CF_ACCOUNT_ID", HEX_ID)
    tunnel: str = _require("CF_TUNNEL_ID", UUID)
    token_file: Path = Path(os.environ.get("CF_TOKEN_FILE", TOKEN_FILE)).expanduser()
    token: str = token_file.read_text().strip()
    if not BEARER_TOKEN.fullmatch(token):
        sys.exit(f"{token_file} must hold only the API token, on one line")
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
