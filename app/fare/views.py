"""Pages and endpoints: the form, the JSON API, the pipeline as JSON, and a health check.

Every prediction is rate limited per client. Errors never show internals: a failure is logged with
its traceback and the client gets a fixed message.
"""

import json
import logging
from datetime import datetime
from typing import Any

from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from fast_inference import EngineError

from fare import serving
from fare.forms import JSONTripError, TripForm, trip_from_json
from fare.ratelimit import WINDOW_SECONDS, limited

logger = logging.getLogger(__name__)

DEFAULT_TRIP: dict[str, Any] = {
    "pickup_zone": 161,  # Midtown Center
    "dropoff_zone": 236,  # Upper East Side North
    "pickup_time": datetime(2026, 8, 12, 9, 15),  # noqa: DTZ001 - New York time
    "trip_distance": "2.40",
}


def _json(data: dict[str, Any], status: int = 200) -> JsonResponse:
    """Answer with strict JSON: a NaN or an infinity fails here instead of reaching a client.

    Args:
        data: The body.
        status: The HTTP status.

    Returns:
        The response.

    """
    return JsonResponse(data, status=status, json_dumps_params={"allow_nan": False})


def _limited_page(request: HttpRequest) -> HttpResponse:
    """Tell a client it is asking too often.

    Args:
        request: The request.

    Returns:
        A 429 page with Retry-After.

    """
    response: HttpResponse = render(request, "fare/limited.html", {"seconds": WINDOW_SECONDS}, status=429)
    response["Retry-After"] = str(WINDOW_SECONDS)
    return response


def _summary(stage: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """Describe in a few words what a pipeline step does and learned.

    Args:
        stage: The step's entry in ``stages_json``.
        state: The step's entry in ``stages_state``.

    Returns:
        Lines for the page.

    """
    kernel: str = stage["kernel"]
    params: dict[str, Any] = stage["params"]
    if kernel == "cast":
        return [f"{column} becomes {target}" for column, target in params["types"].items()]
    if kernel == "label_encode":
        return [
            f"{column}: {len(mapping)} values seen, each with an index; anything else is 0"
            for column, mapping in state.get("label_maps", {}).items()
        ]
    if kernel == "select":
        return ["keeps " + ", ".join(params["columns"])]
    if kernel == "standard_scale":
        return [
            f"{column}: minus the mean {state['means'][column]:.4f}, divided by the standard deviation "
            f"{state['scales'][column]:.4f}"
            for column in state.get("means", {})
        ]
    return [json.dumps(params, sort_keys=True)]


def _pipeline(model: serving.Served) -> list[dict[str, Any]]:
    """Lay the pipeline out for the page.

    Args:
        model: The loaded model.

    Returns:
        One entry per step: its stage, kernel, parameters and summary.

    """
    return [
        {
            "stage": stage["stage"],
            "kernel": stage["kernel"],
            "params": json.dumps(stage["params"], sort_keys=True),
            "summary": _summary(stage, model.states.get(stage["stage"], {})),
        }
        for stage in model.stages
    ]


def _price(model: serving.Served, form: TripForm) -> serving.Prediction:
    """Price a validated trip.

    Args:
        model: The loaded model.
        form: A valid form.

    Returns:
        The prediction.

    """
    data: dict[str, Any] = form.cleaned_data
    return serving.predict(
        model, data["pickup_zone"], data["dropoff_zone"], data["pickup_time"], float(data["trip_distance"])
    )


@require_http_methods(["GET", "POST"])
def index(request: HttpRequest) -> HttpResponse:
    """Show the form; on POST, price the trip and show how.

    Args:
        request: The request.

    Returns:
        The page.

    """
    model: serving.Served | None = serving.served()
    if model is None:
        return render(request, "fare/unavailable.html", status=503)
    prediction: serving.Prediction | None = None
    status: int = 200
    if request.method == "POST":
        if limited(request, "predict", settings.FARE_RATE_LIMIT):
            return _limited_page(request)
        form: TripForm = TripForm(request.POST, zones=model.zones)
        if form.is_valid():
            try:
                prediction = _price(model, form)
            except EngineError, ValueError:
                logger.exception("pricing a trip failed")
                form.add_error(None, "The model could not price this trip.")
                status = 500
        else:
            status = 400
    else:
        form = TripForm(initial=DEFAULT_TRIP, zones=model.zones)
    context: dict[str, Any] = {
        "form": form,
        "prediction": prediction,
        "pipeline": _pipeline(model),
        "info": model.info,
        "zones": model.zones,
    }
    return render(request, "fare/index.html", context, status=status)


@csrf_exempt  # Stateless: there is no cookie or session for another site's request to ride on.
@require_POST
def api_predict(request: HttpRequest) -> JsonResponse:
    """Price a trip sent as JSON.

    Args:
        request: A POST whose body is ``{"pickup_zone", "dropoff_zone", "pickup_time", "trip_distance"}``.

    Returns:
        The fare, the features, the trace and the timings; or an error.

    """
    model: serving.Served | None = serving.served()
    if model is None:
        return _json({"error": "the model is not trained yet"}, status=503)
    if limited(request, "predict", settings.FARE_RATE_LIMIT):
        response: JsonResponse = _json({"error": "too many requests"}, status=429)
        response["Retry-After"] = str(WINDOW_SECONDS)
        return response
    if request.content_type != "application/json":
        return _json({"error": "send application/json"}, status=415)
    try:
        data: dict[str, str] = trip_from_json(request.body)
    except RequestDataTooBig:
        return _json({"error": "the body is too large"}, status=413)
    except JSONTripError as error:
        return _json({"error": str(error)}, status=400)
    form: TripForm = TripForm(data, zones=model.zones)
    if not form.is_valid():
        fields: dict[str, list[str]] = {
            name: [str(message) for message in errors] for name, errors in form.errors.items()
        }
        return _json({"error": "invalid trip", "fields": fields}, status=400)
    try:
        prediction: serving.Prediction = _price(model, form)
    except EngineError, ValueError:
        logger.exception("pricing a trip failed")
        return _json({"error": "the model could not price this trip"}, status=500)
    return _json(
        {
            "fare": round(prediction.fare, 2),
            "features": prediction.features,
            "trace": [
                {"stage": step.stage, "kernel": step.kernel, "output": step.output, "changed": sorted(step.changed)}
                for step in prediction.trace
            ],
            "timings_us": prediction.timings_us,
            "unknown": prediction.unknown,
        }
    )


@require_GET
def api_pipeline(request: HttpRequest) -> JsonResponse:
    """Describe the fitted pipeline: its steps, and what each learned.

    Args:
        request: The request.

    Returns:
        ``{"stages": stages_json, "states": stages_state}``, parsed.

    """
    model: serving.Served | None = serving.served()
    if model is None:
        return _json({"error": "the model is not trained yet"}, status=503)
    return _json({"stages": model.stages, "states": model.states})


@require_GET
def healthz(request: HttpRequest) -> JsonResponse:
    """Report whether the model is loaded.

    Args:
        request: The request.

    Returns:
        ``{"status": "ok"}`` with 200, or ``{"status": "no model"}`` with 503.

    """
    if serving.served() is None:
        return _json({"status": "no model"}, status=503)
    return _json({"status": "ok"})
