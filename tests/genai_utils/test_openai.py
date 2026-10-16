from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
from pytest import mark, param, raises

from genai_utils.llm.openai import NoOutputError, OpenaiChat, _is_model_loading


def _make_response(content):
    """Build a minimal stand-in for an OpenAI chat completion response.

    ``run_message`` reads the text off the object and hands
    ``response.model_dump()`` to ``log_tokens_per_second``, so the stub needs
    both.
    """
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        model_dump=lambda: {"choices": [{"message": {"content": content}}]},
    )


def _mock_client() -> MagicMock:
    """A stand-in for ``AsyncOpenAI`` whose ``create`` can be awaited."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


class _ModelLoadingError(openai.InternalServerError):
    """A 503 InternalServerError without the awkward real constructor."""

    def __init__(self) -> None:
        self.status_code = 503


def test_is_model_loading_false_for_other_exceptions() -> None:
    assert _is_model_loading(ValueError("nope")) is False


def test_is_model_loading_true_for_503() -> None:
    assert _is_model_loading(_ModelLoadingError()) is True


async def test_run_message_retries_while_model_loads() -> None:
    """A 503 (model still loading) should be retried, then succeed."""
    llm = OpenaiChat(model_url="http://localhost:1234", model_name="test-model")
    llm.client = _mock_client()
    llm.client.chat.completions.create.side_effect = [
        _ModelLoadingError(),
        _make_response("ready now"),
    ]

    # patch out tenacity's sleep so the exponential backoff doesn't slow the test
    with patch.object(OpenaiChat.run_message.retry, "sleep", new=AsyncMock()):
        result = await llm.run_message("hello", use_thinking=False)

    assert result == "ready now"
    assert llm.client.chat.completions.create.call_count == 2


async def test_run_message_returns_content() -> None:
    llm = OpenaiChat(model_url="http://localhost:1234", model_name="test-model")
    llm.client = _mock_client()
    llm.client.chat.completions.create.return_value = _make_response("the answer")

    assert await llm.run_message("hello", use_thinking=False) == "the answer"


async def test_run_message_raises_on_no_output() -> None:
    llm = OpenaiChat(model_url="http://localhost:1234", model_name="test-model")
    llm.client = _mock_client()
    llm.client.chat.completions.create.return_value = _make_response(None)

    with raises(NoOutputError):
        await llm.run_message("hello", use_thinking=False)


@mark.parametrize(
    "use_thinking,expected_enabled",
    [
        param(True, True, id="thinking enabled"),
        param(False, False, id="thinking disabled"),
    ],
)
async def test_run_message_passes_thinking_config(
    use_thinking, expected_enabled
) -> None:
    llm = OpenaiChat(model_url="http://localhost:1234", model_name="test-model")
    llm.client = _mock_client()
    llm.client.chat.completions.create.return_value = _make_response("ok")

    await llm.run_message("hello", use_thinking=use_thinking)

    _, kwargs = llm.client.chat.completions.create.call_args
    extra_body = kwargs["extra_body"]
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is expected_enabled
    # timing reporting is always requested; harmless for servers that ignore it
    assert extra_body["timings_per_token"] is True
    # the model name is forwarded to the API
    assert kwargs["model"] == "test-model"


async def test_client_rebuilt_when_event_loop_changes() -> None:
    """A client used on a now-dead loop is replaced rather than reused."""
    llm = OpenaiChat(model_url="http://localhost:1234", model_name="test-model")
    first = llm._get_client()
    llm._loop = MagicMock()  # pretend the client was last used on another loop

    assert llm._get_client() is not first


async def test_aclose_leaves_a_usable_client() -> None:
    llm = OpenaiChat(model_url="http://localhost:1234", model_name="test-model")
    old = llm.client

    await llm.aclose()
    await llm.aclose()  # safe to call twice

    assert llm.client is not old
    assert not llm.client.is_closed()
