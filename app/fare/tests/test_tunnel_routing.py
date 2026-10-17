"""The tunnel's routing: path blocks first for every hostname, the catch-all last, and a push that hides its token."""

import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from deploy.cloudflared import tunnel_routing

PROD: str = "fare.example.dev"
STAGING: str = "fare-staging.example.dev"
TUNNEL: str = "00000000-0000-4000-8000-000000000001"


def _environment(monkeypatch: pytest.MonkeyPatch, token_file: Path, *argv: str) -> None:
    monkeypatch.setenv("FARE_HOST", PROD)
    monkeypatch.setenv("FARE_STAGING_HOST", STAGING)
    monkeypatch.setenv("CF_ACCOUNT_ID", "0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("CF_TUNNEL_ID", TUNNEL)
    monkeypatch.setenv("CF_TOKEN_FILE", str(token_file))
    monkeypatch.setattr(sys, "argv", ["tunnel_routing.py", *argv])


def test_the_path_blocks_come_first_and_the_catch_all_last() -> None:
    rules = tunnel_routing.ingress(PROD, STAGING)
    assert rules[:2] == [
        {"path": "^/healthz", "service": "http_status:404"},
        {"path": "^/admin", "service": "http_status:404"},
    ]
    assert rules[-1] == {"service": "http_status:404"}
    assert {rule["hostname"]: rule["service"] for rule in rules if "hostname" in rule} == {
        PROD: "http://fare-web.fare-web.svc.cluster.local:80",
        STAGING: "http://fare-web.fare-web-staging.svc.cluster.local:80",
    }


@pytest.mark.parametrize(
    "hostname", ["", "fare", "fare.example.dev/x", "-fare.example.dev", "fare example.dev", STAGING]
)
def test_a_malformed_or_repeated_hostname_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hostname: str
) -> None:
    _environment(monkeypatch, tmp_path / "missing")
    monkeypatch.setenv("FARE_HOST", hostname)
    with pytest.raises(SystemExit):
        tunnel_routing.main()


def test_without_push_it_prints_the_routing_without_reading_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _environment(monkeypatch, tmp_path / "missing")
    assert tunnel_routing.main() == 0
    assert json.loads(capsys.readouterr().out) == {"config": {"ingress": tunnel_routing.ingress(PROD, STAGING)}}


def test_a_push_sends_the_routing_and_never_prints_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    token_file: Path = tmp_path / "token"
    token_file.write_text("not-a-real-token-value\n")
    _environment(monkeypatch, token_file, "--push")
    sent: dict[str, Any] = {}

    def urlopen(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        sent.update(
            url=request.full_url,
            method=request.get_method(),
            authorization=request.get_header("Authorization"),
            body=json.loads(bytes(request.data)) if isinstance(request.data, bytes) else None,
        )
        return io.BytesIO(json.dumps({"success": True, "result": {"version": 7}}).encode())

    monkeypatch.setattr(tunnel_routing.urllib.request, "urlopen", urlopen)
    assert tunnel_routing.main() == 0
    assert sent["method"] == "PUT"
    assert sent["url"].endswith(f"/accounts/0123456789abcdef0123456789abcdef/cfd_tunnel/{TUNNEL}/configurations")
    assert sent["authorization"] == "Bearer not-a-real-token-value"
    assert sent["body"] == {"config": {"ingress": tunnel_routing.ingress(PROD, STAGING)}}
    printed = capsys.readouterr()
    assert "not-a-real-token-value" not in printed.out + printed.err
    assert "version 7" in printed.out


def _refuse_any_push(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(*args: object, **kwargs: object) -> None:
        raise AssertionError("pushed")

    monkeypatch.setattr(tunnel_routing.urllib.request, "urlopen", urlopen)


@pytest.mark.parametrize(
    "content",
    [
        "not-a-real-token-value\n# a note\n",
        "# the edit token\nnot-a-real-token-value\n",
        "not-a-real-token-value\rnote",
        'curl -H "Authorization: Bearer not-a-real-token-value"',
    ],
)
def test_a_token_file_holding_more_than_the_token_is_refused_without_printing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    token_file: Path = tmp_path / "token"
    token_file.write_text(content)
    _environment(monkeypatch, token_file, "--push")
    _refuse_any_push(monkeypatch)
    with pytest.raises(SystemExit) as refused:
        tunnel_routing.main()
    printed = capsys.readouterr()
    assert "not-a-real-token-value" not in str(refused.value.code) + printed.out + printed.err


@pytest.mark.parametrize("argument", ["--psuh", "--push=1", "-p", "--dry-run", "--help"])
def test_a_mistyped_flag_or_help_never_pushes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argument: str) -> None:
    token_file: Path = tmp_path / "token"
    token_file.write_text("not-a-real-token-value\n")
    _environment(monkeypatch, token_file, argument)
    _refuse_any_push(monkeypatch)
    with pytest.raises(SystemExit):
        tunnel_routing.main()


def test_the_token_file_defaults_to_a_place_outside_the_repository() -> None:
    # git ignores .cf-token*, but a synced folder would still copy a token kept in the repository.
    repository: Path = Path(tunnel_routing.__file__).resolve().parents[2]
    assert not tunnel_routing.TOKEN_FILE.resolve().is_relative_to(repository)
