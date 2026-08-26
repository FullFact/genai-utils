from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_utils.local_models.serverless_llm import (
    DEFAULT_REQUEST_TIMEOUT,
    NoOutputError,
    ServerlessLLM,
)


def _fake_serverless(response_dict):
    """Build a fake ``Serverless`` client that returns
    ``{"response": response_dict}``.

    Returns the fake ``Serverless`` factory (patch it over
    ``genai_utils.local_models.serverless_llm.Serverless``), the AsyncMock standing in
    for ``endpoint.request``, and the client itself so tests can assert on how the
    client and its ``get_endpoint`` were used.
    """
    request = AsyncMock(return_value={"response": response_dict})
    endpoint = SimpleNamespace(request=request)

    client = MagicMock()
    client.get_endpoint = AsyncMock(return_value=endpoint)
    client.close = AsyncMock()

    serverless = MagicMock(return_value=client)
    return serverless, request, client


def _completion(content):
    return {"choices": [{"message": {"content": content}}]}


async def test_run_message_returns_content() -> None:
    serverless, request, _ = _fake_serverless(_completion("the answer"))
    llm = ServerlessLLM(endpoint_name="ep", model_name="claimants", max_tokens=128)

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        result = await llm.run_message("hello", use_thinking=False)

    assert result == "the answer"

    _, kwargs = request.call_args
    # cost tracks max_tokens so the autoscaler can estimate load
    assert kwargs["cost"] == 128
    assert kwargs["stream"] is False
    payload = request.call_args.args[1]
    assert payload["model"] == "claimants"
    assert payload["max_tokens"] == 128
    assert payload["timings_per_token"] is True


@pytest.mark.parametrize(
    "use_thinking,expected_enabled",
    [
        pytest.param(True, True, id="thinking enabled"),
        pytest.param(False, False, id="thinking disabled"),
    ],
)
async def test_run_message_passes_thinking_config(
    use_thinking, expected_enabled
) -> None:
    serverless, request, _ = _fake_serverless(_completion("ok"))
    llm = ServerlessLLM(endpoint_name="ep")

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        await llm.run_message("hello", use_thinking=use_thinking)

    payload = request.call_args.args[1]
    assert payload["chat_template_kwargs"]["enable_thinking"] is expected_enabled


async def test_run_message_raises_on_no_output() -> None:
    serverless, _, _ = _fake_serverless(_completion(None))
    llm = ServerlessLLM(endpoint_name="ep")

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        with pytest.raises(NoOutputError):
            await llm.run_message("hello", use_thinking=False)


async def test_client_and_endpoint_reused_across_calls() -> None:
    # The whole point of the reuse: constructing the client (SSL cert download +
    # session) and resolving the endpoint (a control-plane round-trip) happen
    # once, not once per request.
    serverless, request, client = _fake_serverless(_completion("ok"))
    llm = ServerlessLLM(endpoint_name="ep")

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        for _ in range(3):
            await llm.run_message("hello", use_thinking=False)

    assert serverless.call_count == 1
    assert client.get_endpoint.call_count == 1
    assert request.call_count == 3


async def test_aclose_closes_client_and_allows_rebuild() -> None:
    serverless, _, client = _fake_serverless(_completion("ok"))
    llm = ServerlessLLM(endpoint_name="ep")

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        await llm.run_message("hello", use_thinking=False)
        await llm.aclose()

        client.close.assert_awaited_once()

        # A subsequent call rebuilds the client rather than using a closed one.
        await llm.run_message("hello again", use_thinking=False)

    assert serverless.call_count == 2


async def test_aclose_is_a_noop_before_any_call() -> None:
    llm = ServerlessLLM(endpoint_name="ep")
    # Should not raise even though no client was ever built.
    await llm.aclose()


async def test_request_carries_a_time_budget() -> None:
    """Without a timeout the SDK's retry loop skips its own time checks and
    retries forever, billing GPU time on every attempt."""
    serverless, request, _ = _fake_serverless(_completion("hi"))
    llm = ServerlessLLM(endpoint_name="ep")

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        await llm.run_message("hello", use_thinking=False)

    assert request.await_args.kwargs["timeout"] == DEFAULT_REQUEST_TIMEOUT


async def test_request_timeout_is_configurable() -> None:
    serverless, request, _ = _fake_serverless(_completion("hi"))
    llm = ServerlessLLM(endpoint_name="ep", request_timeout=42.0)

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        await llm.run_message("hello", use_thinking=False)

    assert request.await_args.kwargs["timeout"] == 42.0
