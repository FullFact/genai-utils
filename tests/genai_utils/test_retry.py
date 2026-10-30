import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import openai
from google.genai import errors
from pytest import LogCaptureFixture, MonkeyPatch, fixture, mark, param, raises
from tenacity import wait_none

from genai_utils.llm import NoOutputError, retry
from genai_utils.llm.gemini import GeminiChat, GeminiError, ModelConfig
from genai_utils.llm.openai import OpenaiChat
from genai_utils.llm.retry import (
    Retryable,
    check_n_attempts,
    retry_on_failure,
    retry_on_failure_sync,
)
from genai_utils.llm.serverless_openai import ServerlessChat

_GEMINI_CONFIG = ModelConfig(project="p", location="l", model_name="m")


@fixture(autouse=True)
def _no_wait(monkeypatch: MonkeyPatch) -> None:
    """Skip the backoff between attempts so the tests run instantly."""
    monkeypatch.setattr(retry, "_WAIT", wait_none())


class _Transient(Exception):
    pass


class _Permanent(Exception):
    pass


class _FakeClient(Retryable):
    """Fails with each error in ``failures`` in turn, then returns "ok"."""

    def __init__(self, n_attempts: int, failures: list[Exception]) -> None:
        self.n_attempts = n_attempts
        self.failures = failures
        self.calls = 0

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, _Transient)

    def _attempt(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"

    @retry_on_failure
    async def run(self) -> str:
        return self._attempt()

    @retry_on_failure_sync
    def run_sync(self) -> str:
        return self._attempt()


async def test_succeeds_on_a_later_attempt() -> None:
    client = _FakeClient(n_attempts=3, failures=[_Transient(), _Transient()])

    assert await client.run() == "ok"
    assert client.calls == 3


async def test_retries_log_monitoring_fields(caplog: LogCaptureFixture) -> None:
    client = _FakeClient(n_attempts=2, failures=[_Transient()])

    await client.run()

    [record] = [r for r in caplog.records if r.name == "genai_utils.llm.retry"]
    assert record.levelname == "WARNING"
    assert record.event_metric == "model_retry"
    assert record.model == "_FakeClient"
    assert record.attempt == 1
    assert record.error_type == "_Transient"


async def test_raises_last_error_once_attempts_run_out() -> None:
    last = _Transient("last")
    client = _FakeClient(n_attempts=2, failures=[_Transient("first"), last])

    with raises(_Transient) as exc_info:
        await client.run()

    assert exc_info.value is last
    assert client.calls == 2


async def test_does_not_retry_errors_that_are_not_retryable() -> None:
    client = _FakeClient(n_attempts=3, failures=[_Permanent()])

    with raises(_Permanent):
        await client.run()

    assert client.calls == 1


async def test_one_attempt_means_no_retries() -> None:
    client = _FakeClient(n_attempts=1, failures=[_Transient()])

    with raises(_Transient):
        await client.run()

    assert client.calls == 1


def test_sync_succeeds_on_a_later_attempt() -> None:
    client = _FakeClient(n_attempts=2, failures=[_Transient()])

    assert client.run_sync() == "ok"
    assert client.calls == 2


def test_sync_does_not_retry_errors_that_are_not_retryable() -> None:
    client = _FakeClient(n_attempts=3, failures=[_Permanent()])

    with raises(_Permanent):
        client.run_sync()

    assert client.calls == 1


@mark.parametrize("n_attempts", [0, -1])
def test_check_n_attempts_rejects_less_than_one(n_attempts: int) -> None:
    with raises(ValueError):
        check_n_attempts(n_attempts)


@mark.parametrize(
    "client_cls,kwargs",
    [
        param(OpenaiChat, {"model_url": "http://x"}, id="Openai"),
        param(ServerlessChat, {"endpoint_name": "ep"}, id="Serverless"),
        param(GeminiChat, {"model_config": _GEMINI_CONFIG}, id="Gemini"),
    ],
)
def test_clients_validate_n_attempts(
    client_cls: type[OpenaiChat | ServerlessChat | GeminiChat], kwargs: dict[str, Any]
) -> None:
    assert client_cls(**kwargs, n_attempts=3).n_attempts == 3
    with raises(ValueError):
        client_cls(**kwargs, n_attempts=0)


def _openai_error(cls: type[openai.APIStatusError], status: int) -> Exception:
    response = MagicMock(status_code=status, headers={})
    return cls("boom", response=response, body=None)


@mark.parametrize(
    "exc,expected",
    [
        param(openai.APIConnectionError(request=MagicMock()), True, id="connection"),
        param(openai.APITimeoutError(request=MagicMock()), True, id="timeout"),
        param(_openai_error(openai.RateLimitError, 429), True, id="429"),
        param(_openai_error(openai.InternalServerError, 500), True, id="500"),
        param(_openai_error(openai.InternalServerError, 503), False, id="503"),
        param(_openai_error(openai.BadRequestError, 400), False, id="400"),
        param(NoOutputError(), True, id="no output"),
        param(ValueError(), False, id="other"),
    ],
)
def test_openai_is_retryable(exc: Exception, expected: bool) -> None:
    assert OpenaiChat(model_url="http://x").is_retryable(exc) is expected


def test_openai_sdk_does_not_retry_on_top_of_ours() -> None:
    assert OpenaiChat(model_url="http://x").client.max_retries == 0


@mark.parametrize(
    "exc,expected",
    [
        param(TimeoutError(), True, id="TimeoutError"),
        param(asyncio.TimeoutError(), True, id="asyncio.TimeoutError"),
        param(NoOutputError(), True, id="no output"),
        param(Exception("Endpoint ep could not be found"), False, id="other"),
    ],
)
def test_serverless_is_retryable(exc: Exception, expected: bool) -> None:
    assert ServerlessChat(endpoint_name="ep").is_retryable(exc) is expected


@mark.parametrize(
    "exc,expected",
    [
        param(errors.ServerError(500, {}), True, id="500"),
        param(errors.ServerError(503, {}), True, id="503"),
        param(errors.ServerError(501, {}), False, id="501"),
        param(errors.ClientError(429, {}), True, id="429"),
        param(errors.ClientError(408, {}), True, id="408"),
        param(errors.ClientError(400, {}), False, id="400"),
        param(httpx.ConnectError("refused"), True, id="httpx connect"),
        param(httpx.ReadTimeout("slow"), True, id="httpx timeout"),
        param(httpx.RemoteProtocolError("dropped"), True, id="httpx protocol"),
        param(aiohttp.ServerDisconnectedError(), True, id="aiohttp disconnect"),
        param(aiohttp.ClientOSError(), True, id="aiohttp os error"),
        param(asyncio.TimeoutError(), True, id="timeout"),
        param(NoOutputError(), False, id="no output"),
        param(GeminiError(), False, id="bad options"),
    ],
)
def test_gemini_is_retryable(exc: Exception, expected: bool) -> None:
    assert GeminiChat(model_config=_GEMINI_CONFIG).is_retryable(exc) is expected


async def test_openai_run_message_retries_transient_errors() -> None:
    llm = OpenaiChat(model_url="http://x", model_name="m", n_attempts=2)
    llm.client = MagicMock()
    llm.client.chat.completions.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=MagicMock())
    )

    with raises(openai.APIConnectionError):
        await llm.run_message("hello", use_thinking=False)

    assert llm.client.chat.completions.create.call_count == 2
