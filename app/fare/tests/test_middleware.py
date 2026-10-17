import logging

import pytest
from django.http import HttpResponse
from django.test import RequestFactory
from pytest_django.fixtures import Settings

from fare.ratelimit import client_ip
from fare.request_log import MAX_FIELD_LEN, RequestLogMiddleware

RF: RequestFactory = RequestFactory()


def _middleware() -> RequestLogMiddleware:
    return RequestLogMiddleware(lambda request: HttpResponse("ok", status=203))


def test_one_record_per_request_with_its_fields(caplog: pytest.LogCaptureFixture) -> None:
    request = RF.post("/api/predict", REMOTE_ADDR="203.0.113.7", HTTP_USER_AGENT="agent-x")
    with caplog.at_level(logging.INFO, logger="fare.request"):
        response = _middleware()(request)
    assert response.status_code == 203
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage().startswith("POST /api/predict 203 ")
    fields = record.__dict__
    assert (fields["method"], fields["path"], fields["status"]) == ("POST", "/api/predict", 203)
    assert (fields["ip"], fields["user_agent"]) == ("203.0.113.7", "agent-x")


def test_client_values_are_cut_and_the_health_check_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="fare.request"):
        _middleware()(RF.get("/" + "p" * 500, HTTP_USER_AGENT="u" * 500))
        _middleware()(RF.get("/healthz"))
    assert len(caplog.records) == 1
    fields = caplog.records[0].__dict__
    assert len(fields["path"]) == MAX_FIELD_LEN
    assert len(fields["user_agent"]) == MAX_FIELD_LEN


def test_forwarded_addresses_count_only_with_trusted_proxies(settings: Settings) -> None:
    forged = RF.get("/", REMOTE_ADDR="10.0.0.2", HTTP_X_FORWARDED_FOR="1.2.3.4, 198.51.100.9")
    settings.RATELIMIT_TRUSTED_FORWARDED_HOPS = 0
    assert client_ip(forged) == "10.0.0.2"  # with no proxy of ours, the header is the client's own claim
    settings.RATELIMIT_TRUSTED_FORWARDED_HOPS = 1
    assert client_ip(forged) == "198.51.100.9"  # the entry our proxy appended, not the client's 1.2.3.4
    garbage = RF.get("/", REMOTE_ADDR="10.0.0.2", HTTP_X_FORWARDED_FOR="not-an-ip")
    assert client_ip(garbage) == "10.0.0.2"


def test_cloudflares_header_counts_only_when_trusted_and_valid(settings: Settings) -> None:
    tunneled = RF.get("/", REMOTE_ADDR="10.42.0.7", HTTP_CF_CONNECTING_IP="198.51.100.9")
    settings.RATELIMIT_TRUST_CF_HEADER = False
    assert client_ip(tunneled) == "10.42.0.7"  # untrusted, the header is just the client's claim
    settings.RATELIMIT_TRUST_CF_HEADER = True
    assert client_ip(tunneled) == "198.51.100.9"  # the visitor, as Cloudflare's edge recorded it
    garbage = RF.get("/", REMOTE_ADDR="10.42.0.7", HTTP_CF_CONNECTING_IP="not-an-ip\nforged")
    assert client_ip(garbage) == "10.42.0.7"
    settings.RATELIMIT_TRUSTED_FORWARDED_HOPS = 1
    both = RF.get("/", REMOTE_ADDR="10.42.0.7", HTTP_CF_CONNECTING_IP="198.51.100.9", HTTP_X_FORWARDED_FOR="1.2.3.4")
    assert client_ip(both) == "198.51.100.9"  # the edge's record wins over a forwarded list


def test_control_characters_cannot_forge_a_log_line(caplog: pytest.LogCaptureFixture) -> None:
    request = RF.get("/", HTTP_USER_AGENT="agent\nforged")
    # A server percent-decodes %0d%0a into the path; RequestFactory would strip them from a URL.
    request.path = "/\r\nINFO fare.request GET /admin 200 0ms"
    with caplog.at_level(logging.INFO, logger="fare.request"):
        _middleware()(request)
    message: str = caplog.records[0].getMessage()
    assert "\n" not in message and "\r" not in message
    assert message.startswith("GET /\\r\\nINFO fare.request")
    assert caplog.records[0].__dict__["user_agent"] == "agent\\nforged"
