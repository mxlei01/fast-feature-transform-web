# fast-feature-transform-web

A small Django site that prices New York City yellow taxi trips with a neural network, and shows
how each trip's features were made. The features come from
[fast-inference](#updating-fast-inference), a feature pipeline library with a C++ core: the same
fitted pipeline that wrote the training data transforms every request.

| | |
| --- | --- |
| Data | NYC TLC yellow taxi trips, August 2026: 2,105,008 of 3,336,716 trips kept |
| Split | 1,643,095 trips from August 1–24 for training, 461,913 from August 25–31 for validation |
| Validation error | **$1.64** mean absolute error, $2.43 root mean squared error, R² 0.964 |
| Baseline | $2.41, $3.29 and R² 0.935 for a straight line from distance alone |
| Request cost | about 1.5 µs in the pipeline and 55 µs in the network while requests keep coming, and ten or more times that for a lone request after a pause, on an M4 Mac mini |

## How it works

**Training** (`training/`):

1. Download one month of [TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
   and keep ordinary metered trips: card or cash, the standard rate, a fare from the $3.00 meter
   drop to $200, 0–100 miles, 1–180 minutes, 1–6 passengers, both zones inside the city.
2. Turn the pickup time into its hour and weekday. A time is carried as an epoch, microseconds since
   1970 counting the New York wall clock, which is how fast-inference reads a naive `datetime`, so
   training and serving compute the same numbers.
3. Fit the fast-inference pipeline on the training days:

   | Step | Kernel | Learns |
   | --- | --- | --- |
   | `categories` | `cast` | nothing: location IDs, hour and weekday become text |
   | `indices` | `label_encode` | an index per value seen at least 10 times; rare and unseen values share index 0 |
   | `distance` | `standard_scale` | the mean and standard deviation of distance; a trip's distance becomes its number of standard deviations from the mean |
   | `features` | `select` | nothing: fixes the output columns and their order |

4. Write both splits through the pipeline straight into memmaps, and train the network on them.
   Each categorical feature gets an embedding of width 8, all kept in one shared table. The
   embeddings and the distance go through hidden layers of 128, 64 and 32 units (each a linear
   layer, batch normalization, ReLU and dropout 0.3) to the fare in dollars, whose starting value
   is the average fare. Adam with learning rate 0.001 minimizes squared error on batches of 4,096
   trips. Training stops once the validation loss has not improved for 15 epochs (at most 300)
   and keeps the best epoch's weights; since the validation days choose that epoch, the validation
   error is a little optimistic.
5. Save everything serving needs in `artifacts/`: the pipeline, the weights, the configuration with
   its metrics, and the zone names. The artifacts are small (about 110 KB) and committed.

**Serving** (`app/`): the model loads once per process, after checking that its pipeline and network
come from the same training run. A request's trip is validated, turned into the same record, run
through `transform_dict` and the network, and answered with the fare, how long each part took, and
the trip after every pipeline step. The page also describes the fitted pipeline from `stages_json()`
and `stages_state()`.

What the model does not cover: the airport flat fares (JFK to Manhattan, Newark and the negotiated
rates are left out of training), and zones with fewer than 10 trips in the training days, which share
one estimate; the page and the API name such values in `unknown`. Distances far from the usual
trips are extrapolated: trips under half a mile can price below the $3.00 meter drop, as low as
$1.44 for 0.1 miles.

**Choices measured.** The distance step and what the network predicts were chosen on the validation
days, with the network and training above:

| Distance step | Network predicts | Seed | Validation MAE | RMSE | R² |
| --- | --- | --- | --- | --- | --- |
| `standard_scale` | dollars | 0 | $1.64 | $2.43 | 0.964 |
| `standard_scale` | dollars | 1 | $1.64 | $2.43 | 0.964 |
| `power_transform` | dollars | 0 | $1.63 | $2.43 | 0.965 |
| `power_transform` | dollars | 1 | $1.66 | $2.44 | 0.964 |
| `standard_scale` | `log1p` of dollars | 0 | $1.67 | $2.64 | 0.958 |
| `power_transform` | `log1p` of dollars | 0 | $1.67 | $2.60 | 0.959 |

Every run that predicts dollars beats every run that predicts `log1p` of the fare on all three
measures, most of all on root mean squared error: squared error on the logarithm counts a dollar off
a long trip for less than a dollar off a short one. The two distance steps trade places between
seeds, so the pipeline keeps the simpler `standard_scale`, whose two seeds also agree more closely.
To reproduce a row, change the `distance` step in `training/features.py` or `TARGET` in
`training/train.py`, and run `uv run python -m training.train` with that `--seed`.

## Requirements

- macOS 27 or later on Apple silicon: the fast-inference wheel in `wheels/` is tagged `macosx_27_0_arm64`.
- [uv](https://docs.astral.sh/uv/), which installs the free-threaded CPython 3.14 that
  `.python-version` names.

```sh
uv sync
make run
```

`make run` serves the site at http://127.0.0.1:8000 with DEBUG on.

## Training

`make train` downloads August 2026's trips, about 60 MB, into `data/` and trains on them;
`make train MONTH=2026-07` uses another month.

```sh
make train
make train MONTH=2026-07
```

`python -m training.train --help` lists the options (data directory, epoch limit, patience, batch
size, threads, seed). Trip data goes to `data/`, which is ignored. The run writes
`artifacts.staging/` and swaps it into `artifacts/` only once everything is written, so a failed or
interrupted run leaves the served model as it was.

## API

`POST /api/predict` takes exactly four fields:

```sh
curl -s http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"pickup_zone": 161, "dropoff_zone": 236, "pickup_time": "2026-08-12T09:15", "trip_distance": 2.4}'
```

| Field | Rule |
| --- | --- |
| `pickup_zone`, `dropoff_zone` | an integer TLC location ID from 1 to 263 that the zone list holds |
| `pickup_time` | a string, exactly `YYYY-MM-DDTHH:MM` or `YYYY-MM-DDTHH:MM:SS`, New York time without an offset, 2009–2099 |
| `trip_distance` | a number of miles from 0.1 to 100 with at most two decimal places |

The answer holds `fare` (the metered fare in dollars, before tips, tolls and surcharges), `features`
(what the network received), `trace` (each step's stage, kernel, output record and changed keys),
`timings_us`, and `unknown` (the features whose value was rare or unseen in training). A request that is not exactly such an object gets a 400 with a fixed message, a body
over 4 KB gets 413, any other content type gets 415, and too many requests get 429.

`GET /api/pipeline` returns `{"stages": stages_json, "states": stages_state}`, and `GET /healthz`
reports whether the model is loaded; it answers over plain HTTP too, for probes that bypass the TLS
proxy.

## Deploying

`make serve` runs gunicorn on 127.0.0.1:8000 with DEBUG off: one process, so one copy of the model and one
rate-limit budget per client, with eight threads that run in parallel on free-threaded Python. It needs:

| Variable | Meaning |
| --- | --- |
| `DJANGO_SECRET_KEY` | required; for example `python -c "import secrets; print(secrets.token_urlsafe(50))"` |
| `DJANGO_ALLOWED_HOSTS` | comma-separated host names (default `localhost,127.0.0.1`) |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | the site's origin, such as `https://fare.example.com`, behind a proxy |
| `DJANGO_SSL_REDIRECT` | `false` when no TLS proxy is in front (default `true`) |
| `RATELIMIT_TRUSTED_FORWARDED_HOPS` | how many proxies in front append `X-Forwarded-For` (default 0) |
| `RATELIMIT_TRUST_CF_HEADER` | `true` behind Cloudflare: take the client's address from `CF-Connecting-IP` (default `false`) |
| `PROBE_BAN_SECONDS` | how long an address that asked for a scanner's path is refused (default 3600) |
| `FARE_ARTIFACTS_DIR` | another trained model (default `artifacts/`) |
| `FARE_RATE_LIMIT` | predictions per client per minute (default 30), counted per server process |
| `DJANGO_HSTS_SECONDS`, `DJANGO_HSTS_SUBDOMAINS`, `DJANGO_HSTS_PRELOAD` | HSTS over HTTPS (one year; off; off) |
| `LOG_FORMAT`, `LOG_LEVEL` | `json` or `plain` (default `json` with DEBUG off), and the level |

For example, locally without TLS:

```sh
DJANGO_SECRET_KEY=... DJANGO_SSL_REDIRECT=false make serve
```

The `Dockerfile` packages the site as an image. [deploy/README.md](deploy/README.md) covers the
Kubernetes manifests, the ArgoCD flow, and what is still missing before the site can go live.

## Security

Every value a client sends goes through one Django form (`app/fare/forms.py`), the page's and the
API's alike:

- **Zones** must be one of the listed zone IDs.
- **Times** must be in the one accepted format, with no offset, and fall inside 2009–2099.
- **Distances** must be a decimal number in range: `nan`, `inf`, exponents beyond the range and
  extra decimal places are refused.
- **The API's JSON** must be an object with exactly the four keys and the right types. A bool is not
  an integer, and `NaN`/`Infinity`, which Python's `json` would accept, are refused. Error messages
  never repeat what the client sent.
- **Request bodies** are capped at 4 KB.
- **Templates** escape everything, and the Content Security Policy allows no script at all.
- **Other headers:** forms may only post back to the site, and no site may frame it. Responses
  also set `nosniff`, a same-origin referrer policy and a Permissions Policy.
- **CSRF:** the form needs Django's token (an HttpOnly, SameSite=Strict cookie). The API is
  stateless and has no cookie to ride on.
- **Rate limits:** predictions are limited per client address, and per /64 for IPv6. `X-Forwarded-For`
  is believed only for the proxies you declare, and Cloudflare's `CF-Connecting-IP` only when you say
  the site is behind Cloudflare.
- **Scanners:** a request for a path only a vulnerability scanner would ask for (any `.php` URL,
  WordPress, `.env`, `.git` and the others in `app/fare/probe_guard.py`) gets a 404, and its sender's
  address (for IPv6, its whole /64) is refused for an hour. Current browsers mark their requests with
  `Sec-Fetch-Site`, and those ban no one, so a hostile page or a link in an email cannot get a visitor
  banned; older browsers send no such header. A private address, such as the tunnel connector's, is
  never banned. Behind a proxy with a public address, declare it as above, or one probe through it bans
  everyone behind it.
- **No stored state:** there is no database, no session and no login.
- **Failures** are logged with their traceback, and the client gets a fixed message.
- **Model loading:** the weights load with `torch.load(..., weights_only=True)`, which reads
  tensors and never runs pickled code.

## Logging

Every request except the health check is logged once by `fare.request` with its method, path,
status, duration, client address and user agent. Values that come from the client are cut to 200 characters, with control
characters escaped so a request cannot forge a log line. With `LOG_FORMAT=json` each record is
one JSON object; the plain format reads `POST /api/predict 200 3ms`. Training logs the same way.

## Development

`make all` formats, lints, type-checks, tests and runs bandit; `make pytest` runs only the tests,
which train a small model on synthetic trips with no download.

```sh
make all
make pytest
```

CI runs ruff and bandit, and renders the Kubernetes manifests. The tests and the type check need the
macOS-only wheel, so they run locally.

## Updating fast-inference

The wheel in `wheels/` is the fast-inference library, built from its repository
(mxlei01/fast-feature-transform, which is private) with `uv build --wheel` for free-threaded CPython
3.14 on macOS 27 arm64. To update it, copy the new wheel into `wheels/`, point
`[tool.uv.sources]` in `pyproject.toml` at it, run `uv lock && uv sync`, and retrain so the
artifacts record the version that fitted them.

## Data and license

Trip records and zone names are published by the NYC Taxi and Limousine Commission. The code is
MIT licensed; see `LICENSE`.
