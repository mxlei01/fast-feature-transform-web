# Deploying the site to the cluster

The site deploys the way the cluster's other services do. Each commit gets one image, pinned in
kustomize overlays that ArgoCD syncs. The issue branch under review runs in a staging namespace, gated
by a smoke check, and master runs in prod. Nothing here is live yet: the last section lists what is
still missing.

## Layout

| Path | What it holds |
| --- | --- |
| `../Dockerfile` | The image: Django under gunicorn (one process, eight threads), running as uid 1000, with the trained model, the feature code it imports, and torch's CPU-only build on Linux |
| `manifests/base/` | The Deployment, the Service (port 80 to the pod's 8000), the ConfigMap `fare-web-config`, and `smoke_check.py` |
| `manifests/overlays/staging/` | Namespace `fare-web-staging`, plus the PostSync Job that runs the smoke checks |
| `manifests/overlays/prod/` | Namespace `fare-web` |
| `argocd/` | The two ArgoCD Applications, applied once |

The pod runs one replica, because the rate limit is counted per process. It runs as a non-root
user on a read-only root filesystem, with an `emptyDir` at `/tmp`. The site refuses file uploads before
spooling them, so the `emptyDir` cannot be filled that way. Its probes send `Host: localhost` to
`/healthz`, which is exempt from the HTTPS redirect. The startup probe allows three minutes for the
first request, which loads torch and the model. On macOS the worker holds 247 MB once the model is
loaded; the pod requests 512Mi and is limited to 1Gi.

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
3. `kubectl apply -f deploy/argocd/application-staging.yaml -f deploy/argocd/application-prod.yaml`.
4. Right after applying them, give the live Applications the registry (see "Before the first image"
   below). Until then their pods cannot pull the image.

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

## Still missing

### The public hostname (Cloudflare)

- DNS routes from two hostnames to the cluster's Cloudflare tunnel, one for prod and one for staging,
  for example subdomains of `fast-feature-transform.dev`, which leaves the bare domain for documentation.
- Tunnel hostname rules that send them to `http://fare-web.fare-web.svc.cluster.local:80` and
  `http://fare-web.fare-web-staging.svc.cluster.local:80`. The rules can go in the existing tunnel's
  configuration, or in a tunnel and connector Deployment of this site's own, with its credentials
  Secret.
- Each hostname in its overlay's `fare-web-config`: in `DJANGO_ALLOWED_HOSTS`, and as `https://<hostname>`
  in `DJANGO_CSRF_TRUSTED_ORIGINS`. Keep `localhost` in the list for the probes and the smoke check.
- The visitor's address. Behind the tunnel, every request comes from the connector, so the rate limit
  would put all visitors in one bucket. Either let the rate limiter trust Cloudflare's
  `CF-Connecting-IP` header, which it has no option for yet, or set `RATELIMIT_TRUSTED_FORWARDED_HOPS`
  to match the proxies that append `X-Forwarded-For`.
- Optionally, a rule at the edge that blocks `/healthz` from the internet; the probes stay inside the
  cluster.

### Ingress and TLS

- There is no Ingress, so the Service is reachable only inside the cluster. A LAN Ingress (traefik,
  with a certificate from the cluster's internal CA) would give the site an internal hostname. If the
  rate limiter ever trusts `CF-Connecting-IP`, that Ingress must strip the header so that a LAN client
  cannot forge it.
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

### Cluster policy and logs

- A NetworkPolicy. It should allow TCP 8000 into the pod from four sources:
  - the smoke Job in the same namespace;
  - the tunnel connector, once the hostname is routed;
  - traefik, if an Ingress is added;
  - the nodes' own addresses, so that kubelet probes keep working.

  The node addresses are specific to the cluster, so the policy is not committed here.
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
