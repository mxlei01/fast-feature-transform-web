# Deploying the site to the cluster

The site deploys the way the cluster's other services do. Each commit gets one image, pinned in
kustomize overlays that ArgoCD syncs. The issue branch under review runs in a staging namespace, gated
by a smoke check, and master runs in prod. Nothing here is live yet: the last section lists what is
still missing.

## Layout

| Path | What it holds |
| --- | --- |
| `../Dockerfile` | The image: Django under gunicorn (one process, eight threads), running as uid 1000, with the trained model, the feature code it imports, and torch's CPU-only build on Linux |
| `manifests/base/` | The Deployment, the Service (port 80 to the pod's 8000), the ConfigMap `fare-web-config`, the NetworkPolicy `fare-web`, the traefik middleware `fare-web-strip-cf-headers`, and `smoke_check.py` |
| `manifests/overlays/staging/` | Namespace `fare-web-staging`, plus the PostSync Job that runs the smoke checks |
| `manifests/overlays/prod/` | Namespace `fare-web`, plus the tunnel connector |
| `manifests/cloudflared/` | The Cloudflare Tunnel connector and its NetworkPolicy: two replicas, included by prod only, since one tunnel serves both hostnames |
| `cloudflared/tunnel_routing.py` | The tunnel's routing, pushed to Cloudflare with `make tunnel-routing ARGS=--push` |
| `argocd/` | The ArgoCD project and the two Applications, applied once |

The pod runs one replica, because the rate limit is counted per process. It runs as a non-root
user on a read-only root filesystem, with an `emptyDir` at `/tmp`. The site refuses file uploads before
spooling them, so the `emptyDir` cannot be filled that way. Its probes send `Host: localhost` to
`/healthz`, which is exempt from the HTTPS redirect. The startup probe allows three minutes for the
first request, which loads torch and the model. On macOS the worker holds 247 MB once the model is
loaded; the pod requests 512Mi and is limited to 1Gi. The pod has no service-account token, and its
NetworkPolicy allows it no outgoing connections, not even DNS: the site calls nothing.

## One-time setup

1. Create the Secret in each namespace. It holds only `DJANGO_SECRET_KEY`:

   ```sh
   for ns in fare-web-staging fare-web; do
     kubectl create namespace "$ns"
     kubectl -n "$ns" create secret generic fare-web-secret \
       --from-literal=DJANGO_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')"
   done
   ```

2. Build, push and pin a first image (steps 2 and 3 below), and merge the pins to master. Until then,
   both overlays name the tag `dev`, which no build produces.
3. `kubectl apply -f deploy/argocd/project.yaml`, then
   `kubectl apply -f deploy/argocd/application-staging.yaml -f deploy/argocd/application-prod.yaml`. The
   project allows the Applications only this repository, the two namespaces and the kinds of objects the
   overlays hold. A new kind in the overlays fails the sync until it is added to `project.yaml` and the
   project is applied again by hand.
4. Right after applying them, give the live Applications the registry (see "Before the first image"
   below). Until then their pods cannot pull the image.
5. The tunnel. Copy its token from Cloudflare's dashboard (the `eyJ…` string in the tunnel's connector
   install command) and store it as a Secret in the prod namespace, without writing it to a file. Run
   `read -rs TOKEN` by itself, paste the token and press Enter (nothing is echoed), then:

   ```sh
   printf %s "$TOKEN" | kubectl -n fare-web create secret generic fare-web-cloudflared --from-file=token=/dev/stdin
   unset TOKEN
   ```

   Point each hostname at the tunnel with a proxied CNAME to `<tunnel ID>.cfargotunnel.com`, added on the
   DNS records page of `fast-feature-transform.dev`; the tunnel's ID is on its page in the dashboard. Do
   not use `cloudflared tunnel route dns`, which writes into the zone its `~/.cloudflared/cert.pem` was
   issued for, whatever the hostname. Then push the routing, and check that the connector answers
   `/healthz` with a 404:

   ```sh
   CF_ACCOUNT_ID=… CF_TUNNEL_ID=… FARE_HOST=<prod hostname> FARE_STAGING_HOST=<staging hostname> make tunnel-routing ARGS=--push
   curl -sI https://<prod hostname>/healthz
   ```

   The push reads an API token with Account, Cloudflare Tunnel, Edit from
   `~/.config/cloudflare/edit-token` (or `CF_TOKEN_FILE`). Keep it outside the repository, which may sit in
   a synced folder, and give it an expiry date: it can read the token of every tunnel in the account.

## A change, end to end

1. Commit the change on its issue branch.
2. `make image` builds `fare-web:<commit>` for arm64 and refuses a tree with uncommitted changes.
   `make push REGISTRY=host:port` pushes it, but only from a host that can reach the registry and has
   run `docker login` against it. The workstation cannot reach the registry, so the image goes through
   the same relay host the cluster's other images use: `docker save` it there, `docker load` it, and
   push it from that host. If docker hangs on the macOS credential helper, run it with a throwaway
   `DOCKER_CONFIG` directory.
3. `make pin` writes that tag into both overlays. Commit the pins on the same branch.
4. Push the branch to origin under its issue number, then run `make deploy-staging`. ArgoCD syncs
   staging and runs the smoke Job, and a failed Job marks the sync failed.
5. Open the pull request. Merging it to master deploys prod.

A change to the manifests alone skips steps 2 and 3. `make manifests` renders both overlays as ArgoCD
will. `make smoke SMOKE_URL=http://host:port` runs the smoke checks against gunicorn itself, for example
through `kubectl port-forward`. It sends `Host: localhost`, which only the pod accepts until the public
hostname is configured.

To roll back, revert the pin commit and merge the revert; ArgoCD then rolls back to the previous image.

## Public access (Cloudflare Tunnel)

No port is open to the internet. The connector in `manifests/cloudflared/` dials out to Cloudflare, whose
edge ends public TLS and sends each request through the tunnel to the connector, which passes it to the
site's Service: prod's in `fare-web`, staging's across namespaces in `fare-web-staging`.

- **Routing.** The tunnel was created in Cloudflare's dashboard, so Cloudflare stores its routing and sends
  it to the connector, which needs only its token. The routing is written down in
  `cloudflared/tunnel_routing.py`; edit it there, check it with `make tunnel-routing`, which only prints
  it, and push it with `make tunnel-routing ARGS=--push`. Its first two rules have the connector answer
  `/healthz`, and any `/admin` path without a dot, with a 404 for every hostname, so those requests never
  reach the site's pods. Only a WAF rule on the zone would stop them at the edge.
- **The visitor's address.** The edge records it in `CF-Connecting-IP`, overwriting any value the visitor
  sent, and the site keys the rate limit, the request log and the scanner ban on it
  (`RATELIMIT_TRUST_CF_HEADER`). The limit and the ban treat an IPv6 visitor's whole /64 as one. Two zone
  settings must stay at their default, off, because each takes the visitor's address out of the header:
  the managed transform "Remove visitor IP headers", after which every visitor counts as a connector and
  no one is banned, and Pseudo IPv4 set to "Overwrite Headers", which puts a made-up IPv4 address there for
  each IPv6 visitor, and such an address is never banned ("Add Header" is harmless). The
  NetworkPolicies let only the connector and the smoke Job reach the site's pods over the pod network, so
  no other pod there can set the header. No policy can block a pod's own node, so a process or a
  hostNetwork pod (a node exporter, say) on that node still can. Any LAN Ingress must use the
  `fare-web-strip-cf-headers` middleware.
- **Scanners.** A request for a path only a scanner would ask for (any `.php` URL, WordPress, `.env`,
  `.git` and the others in `app/fare/probe_guard.py`) gets a 404, and its sender is refused for an hour
  (`PROBE_BAN_SECONDS`), static files aside. The ban lives in the pod; lasting, cross-address blocks
  belong in the zone's security rules at Cloudflare.
- **Kill switch.** `kubectl -n fare-web scale deploy/fare-web-cloudflared --replicas=0` takes the public
  site, both hostnames, down in seconds. It holds until the next sync from git (`selfHeal` is off); to make
  it last, commit `replicas: 0`. For load or abuse, Cloudflare's "Under Attack" mode challenges visitors
  while the site stays up.

## Still missing

### The public hostnames (Cloudflare)

- The two hostnames, one for prod and one for staging, for example subdomains of
  `fast-feature-transform.dev`, which leaves the bare domain for documentation. With them chosen, step 5 of
  the one-time setup routes them.
- Each hostname in its overlay's `fare-web-config`: in `DJANGO_ALLOWED_HOSTS`, and as `https://<hostname>`
  in `DJANGO_CSRF_TRUSTED_ORIGINS`. Keep `localhost` in the list for the probes and the smoke check.
- The zone's own bot and security settings at Cloudflare, which live in its dashboard, not here. Among
  them, a WAF custom rule that blocks `/healthz`, `/admin` and the scanner paths at the edge, and URL
  normalization's "Normalize URLs to origin", so the site receives the path Cloudflare caches under.

### Ingress and TLS

- There is no Ingress, so apart from its own node and `kubectl port-forward`, the site is reachable only
  through the tunnel. A LAN Ingress (traefik, with a certificate from the cluster's internal CA) would
  give it an internal hostname. It would need a rule for traefik in `manifests/base/network-policy.yaml`,
  and must use the `fare-web-strip-cf-headers` middleware, since the site trusts `CF-Connecting-IP`.
- Public TLS ends at Cloudflare's edge. Browsers require HTTPS on every `.dev` name, which the edge
  certificate satisfies.

### Before the first image

- The library for Linux. The site uses a vendored macOS arm64 wheel of fast-inference, which `uv sync`
  in the image cannot install. The library is to be published on PyPI as `fast-feature-transform`, with
  wheels for free-threaded CPython 3.14 on Linux arm64, Linux x86_64 and macOS arm64, and an import
  package renamed to `fast_feature_transform`. Once it is published, replace the vendored wheel and
  its `[tool.uv.sources]` entry with a dependency on it, update the imports, and run `uv lock`.
- The registry. `newName` in both overlays stays the placeholder `registry.invalid/fare-web` in git,
  because this repository is public and the registry's address is a LAN address. Set it only on the
  live Applications, through ArgoCD's `spec.source.kustomize.images`, and never in `deploy/argocd/`.
  Nodes pull with their own registry configuration, so the manifests carry no image pull Secret.
- Measure the pod's memory on a node. The 247 MB above was measured on macOS.

### Secrets, logs and CI

- The Secret in git. `fare-web-secret` is created by hand. To encrypt it with sops and generate it with
  ksops, as the cluster's other services do, it needs the cluster's age recipient.
- Log collection. The site's own records are one JSON object per line on stderr, in the same shape as
  the cluster's other services. gunicorn's start-up and error lines are plain text on the same stream.
  The cluster's log collector ships only the namespaces it selects, so its filter needs `fare-web` and
  `fare-web-staging`. Keeping their records apart from the other services' takes an index of their own
  in the collector's sink, and a matching index pattern in Grafana.
- Image builds in CI. CI builds no image, and images are built from a workstation as the cluster's
  other images are. Once the library is on PyPI, a CI build becomes possible, but pushing would still
  need a runner that can reach the registry.
