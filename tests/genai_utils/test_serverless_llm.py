from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_utils.local_models.serverless_llm import NoOutputError, ServerlessLLM


def _fake_serverless(response_dict):
    """Build a fake ``Serverless`` client that returns
    ``{"response": response_dict}``.

    Returns the fake ``Serverless`` (patch it over
    ``genai_utils.local_models.serverless_llm.Serverless``) and the AsyncMock standing in
    for ``endpoint.request`` so tests can assert on how it was called.
    """
    request = AsyncMock(return_value={"response": response_dict})
    endpoint = SimpleNamespace(request=request)

    client = MagicMock()
    client.get_endpoint = AsyncMock(return_value=endpoint)
    # ``async with Serverless() as client`` -> yields ``client``
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=client), request


def _completion(content):
    return {"choices": [{"message": {"content": content}}]}


async def test_run_message_returns_content() -> None:
    serverless, request = _fake_serverless(_completion("the answer"))
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
    serverless, request = _fake_serverless(_completion("ok"))
    llm = ServerlessLLM(endpoint_name="ep")

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        await llm.run_message("hello", use_thinking=use_thinking)

    payload = request.call_args.args[1]
    assert payload["chat_template_kwargs"]["enable_thinking"] is expected_enabled


async def test_run_message_raises_on_no_output() -> None:
    serverless, _ = _fake_serverless(_completion(None))
    llm = ServerlessLLM(endpoint_name="ep")

    with patch("genai_utils.local_models.serverless_llm.Serverless", serverless):
        with pytest.raises(NoOutputError):
            await llm.run_message("hello", use_thinking=False)
