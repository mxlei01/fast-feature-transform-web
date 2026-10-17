import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.template.defaultfilters import floatformat
from django.test import Client
from pytest_django.fixtures import Settings

from fare.tests.conftest import HOSTILE_ZONE

TRIP: dict[str, Any] = {
    "pickup_zone": 161,
    "dropoff_zone": 236,
    "pickup_time": "2026-08-12T09:15",
    "trip_distance": 2.4,
}


def _post_json(client: Client, body: Any, **extra: Any) -> Any:
    return client.post("/api/predict", data=body, content_type="application/json", **extra)


def test_the_page_shows_the_form_the_pipeline_and_escapes_zone_names(client: Client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    page: str = response.content.decode()
    assert 'name="pickup_zone"' in page and 'name="trip_distance"' in page
    for stage in ("categories", "indices", "distance", "features"):
        assert stage in page
    assert HOSTILE_ZONE not in page
    assert "&lt;script&gt;alert(&quot;zone&quot;)&lt;/script&gt;" in page


def test_the_page_shows_the_models_figures(client: Client, served_artifacts: Path) -> None:
    info: dict[str, Any] = json.loads((served_artifacts / "model.json").read_text())
    page: str = client.get("/").content.decode()
    valid: dict[str, float] = info["valid_metrics"]
    baseline: dict[str, float] = info["baseline_valid_metrics"]
    assert (
        f"${floatformat(valid['mae'], 2)} mean absolute error, ${floatformat(valid['rmse'], 2)} root mean squared"
        f" error, R² {floatformat(valid['r2'], 3)}"
    ) in page
    assert (
        f"${floatformat(baseline['mae'], 2)}, ${floatformat(baseline['rmse'], 2)} and R²"
        f" {floatformat(baseline['r2'], 3)} for a straight line"
    ) in page
    assert f"Embeddings of width {info['config']['embedding_dim']} for " in page
    assert (
        f"dropout {info['config']['dropout']}; kept from epoch {info['best_epoch']} of {len(info['history'])}" in page
    )


def test_the_page_names_the_epoch_kept_and_the_epochs_run(
    client: Client, settings: Settings, served_artifacts: Path, tmp_path: Path
) -> None:
    longer: Path = tmp_path / "longer"
    shutil.copytree(served_artifacts, longer)
    info: dict[str, Any] = json.loads((longer / "model.json").read_text())
    info["history"] = [{**info["history"][0], "epoch": epoch} for epoch in (1, 2, 3)]
    info["best_epoch"] = 2
    (longer / "model.json").write_text(json.dumps(info))
    settings.FARE_ARTIFACTS_DIR = longer
    assert "kept from epoch 2 of 3" in client.get("/").content.decode()


def test_posting_a_trip_prices_it_and_shows_each_step(client: Client) -> None:
    response = client.post("/", {**TRIP, "trip_distance": "2.40"})
    assert response.status_code == 200
    page: str = response.content.decode()
    assert "Estimated fare" in page
    assert "label_encode" in page and "standard_scale" in page
    assert "minus the mean" in page
    assert page.count('class="changed"') >= 4


def test_an_invalid_trip_is_refused_without_repeating_the_values(client: Client) -> None:
    response = client.post(
        "/", {"pickup_zone": "999", "dropoff_zone": "264", "pickup_time": "yesterday", "trip_distance": "nan"}
    )
    assert response.status_code == 400
    page: str = response.content.decode()
    assert page.count("Choose one of the zones.") == 2
    assert "Enter a date and time such as 2026-08-14T08:30." in page
    assert "Estimated fare" not in page


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trip_distance", "0.05"),
        ("trip_distance", "100.01"),
        ("trip_distance", "1.234"),
        ("trip_distance", "inf"),
        ("trip_distance", "1e9"),
        ("trip_distance", "<b>5</b>"),
        ("pickup_time", "2008-12-31T23:59"),
        ("pickup_time", "2100-01-01T00:00"),
        # Only the documented format: no offsets, no date alone, no other ISO spellings.
        ("pickup_time", "2026-08-12T09:15Z"),
        ("pickup_time", "2026-08-12T09:15+00:00"),
        ("pickup_time", "2026-08-12"),
        ("pickup_time", "20260812T0915"),
        ("pickup_time", "2026-W33-3T09:15"),
        ("pickup_time", "2026-08-12 09:15"),
        ("pickup_zone", "0"),
        ("pickup_zone", "132 OR 1=1"),
    ],
)
def test_each_field_refuses_values_outside_its_rules(client: Client, field: str, value: str) -> None:
    response = client.post("/", {**TRIP, field: value})
    assert response.status_code == 400


def test_the_form_needs_its_csrf_token() -> None:
    response = Client(enforce_csrf_checks=True).post("/", TRIP)
    assert response.status_code == 403


def test_the_api_prices_a_trip(client: Client) -> None:
    response = _post_json(client, TRIP)
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert math.isfinite(body["fare"]) and body["fare"] > 0
    assert [step["stage"] for step in body["trace"]] == ["categories", "indices", "distance", "features"]
    assert body["trace"][-1]["output"] == body["features"]
    assert set(body["timings_us"]) == {"record", "pipeline", "network"}
    assert isinstance(body["unknown"], list)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("{", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        (json.dumps({**TRIP, "extra": 1}), "expected exactly the keys"),
        (json.dumps({k: v for k, v in TRIP.items() if k != "trip_distance"}), "expected exactly the keys"),
        (json.dumps({**TRIP, "pickup_zone": True}), "pickup_zone must be an integer"),
        (json.dumps({**TRIP, "pickup_zone": "132"}), "pickup_zone must be an integer"),
        (json.dumps({**TRIP, "dropoff_zone": 230.0}), "dropoff_zone must be an integer"),
        (json.dumps({**TRIP, "pickup_time": 1_786_725_000}), "pickup_time must be a string"),
        (json.dumps({**TRIP, "trip_distance": "17.5"}), "trip_distance must be a number"),
        (json.dumps({**TRIP, "trip_distance": None}), "trip_distance must be a number"),
        (json.dumps({**TRIP, "pickup_time": "2026-08-12T09:15Z"}), "invalid trip"),
        (json.dumps({**TRIP, "pickup_time": "2026-08-12T09:15+05:00"}), "invalid trip"),
        ('{"pickup_zone": 161, "dropoff_zone": 236, "pickup_time": "2026-08-12T09:15", "trip_distance": NaN}', "NaN"),
        (
            '{"pickup_zone": 161, "dropoff_zone": 236, "pickup_time": "2026-08-12T09:15", "trip_distance": Infinity}',
            "Infinity",
        ),
    ],
)
def test_the_api_refuses_json_that_is_not_exactly_a_trip(client: Client, body: str, message: str) -> None:
    response = _post_json(client, body)
    assert response.status_code == 400
    assert message in response.json()["error"]


def test_the_api_validates_values_like_the_form(client: Client) -> None:
    response = _post_json(client, {**TRIP, "dropoff_zone": 264, "trip_distance": 250})
    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid trip",
        "fields": {
            "dropoff_zone": ["Choose one of the zones."],
            "trip_distance": ["Ensure this value is less than or equal to 100."],
        },
    }
    huge: str = '{"pickup_zone": 161, "dropoff_zone": 236, "pickup_time": "2026-08-12T09:15", "trip_distance": 1e999}'
    assert _post_json(client, huge).status_code == 400


def test_a_file_upload_is_refused_before_it_is_spooled() -> None:
    # With a well-formed CSRF cookie the CSRF check reads the body to look for the token. The first
    # file part must be refused there (400), not written to /tmp and then refused for its token (403).
    client = Client(enforce_csrf_checks=True)
    client.cookies["csrftoken"] = "a" * 32
    upload = SimpleUploadedFile("big.bin", b"x" * 50_000)
    assert client.post("/", data={"pickup_zone": "161", "file": upload}).status_code == 400


def test_the_api_checks_type_size_and_method(client: Client) -> None:
    assert client.post("/api/predict", data=json.dumps(TRIP), content_type="text/plain").status_code == 415
    assert client.post("/api/predict", data=TRIP).status_code == 415
    padded: str = json.dumps(TRIP)[:-1] + ', "pad": "' + "x" * 5_000 + '"}'
    assert _post_json(client, padded).status_code == 413
    assert client.get("/api/predict").status_code == 405


def test_the_pipeline_endpoint_returns_stages_json_and_stages_state(client: Client) -> None:
    body: dict[str, Any] = client.get("/api/pipeline").json()
    assert [stage["stage"] for stage in body["stages"]] == ["categories", "indices", "distance", "features"]
    assert list(body["states"]) == ["categories", "indices", "distance", "features"]
    assert set(body["states"]["indices"]["label_maps"]) == {
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_weekday",
    }


def test_without_a_model_the_site_says_so(client: Client, settings: Settings, tmp_path: Path) -> None:
    settings.FARE_ARTIFACTS_DIR = tmp_path
    assert client.get("/").status_code == 503
    assert client.get("/healthz").json() == {"status": "no model"}
    assert _post_json(client, TRIP).status_code == 503


def test_health_check(client: Client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predictions_are_rate_limited_per_client(client: Client, settings: Settings) -> None:
    settings.FARE_RATE_LIMIT = 2
    assert [_post_json(client, TRIP).status_code for _ in range(2)] == [200, 200]
    limited = _post_json(client, TRIP)
    assert limited.status_code == 429
    assert limited["Retry-After"] == "60"
    assert client.post("/", TRIP).status_code == 429
    assert limited["Cache-Control"] == "no-store"
    # Another client has its own budget.
    assert _post_json(client, TRIP, REMOTE_ADDR="198.51.100.4").status_code == 200


@pytest.mark.parametrize(("method", "path"), [("get", "/"), ("get", "/nowhere"), ("put", "/"), ("get", "/healthz")])
def test_every_response_carries_the_security_headers(client: Client, method: str, path: str) -> None:
    response = getattr(client, method)(path)
    assert "script-src" not in response["Content-Security-Policy"]
    assert response["Content-Security-Policy"].startswith("default-src 'none'")
    assert "frame-ancestors 'none'" in response["Content-Security-Policy"]
    assert response["Permissions-Policy"].startswith("camera=()")
    assert response["X-Content-Type-Options"] == "nosniff"
    assert response["X-Frame-Options"] == "DENY"


@pytest.mark.parametrize("path", ["/nowhere", "/missing.css", "/static/fare/./app.css", "/static/fare/../fare/app.css"])
def test_no_error_can_be_cached(client: Client, path: str) -> None:
    # Cloudflare keeps a 404 with no Cache-Control under a .css URL, and may file the dot-segment spellings,
    # which reach the site unchanged and get a 404, under the stylesheet's own URL.
    response = client.get(path)
    assert response.status_code == 404
    assert response["Cache-Control"] == "no-store"
    assert "Cache-Control" not in client.get("/")
