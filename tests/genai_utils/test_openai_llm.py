import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
from pytest import mark, param, raises

from genai_utils.openai_llm import (
    LLM,
    NoOutputError,
    _is_model_loading,
    log_tokens_per_second,
)

LOGGER_NAME = "genai_utils.openai_llm"


def _make_response(content):
    """Build a minimal stand-in for an OpenAI chat completion response."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model_extra={}, usage=None)


class _ModelLoadingError(openai.InternalServerError):
    """A 503 InternalServerError without the awkward real constructor."""

    def __init__(self) -> None:
        self.status_code = 503


def test_is_model_loading_false_for_other_exceptions() -> None:
    assert _is_model_loading(ValueError("nope")) is False


def test_is_model_loading_true_for_503() -> None:
    assert _is_model_loading(_ModelLoadingError()) is True


def test_run_message_retries_while_model_loads() -> None:
    """A 503 (model still loading) should be retried, then succeed."""
    llm = LLM(model_url="http://localhost:1234", model_name="test-model")
    llm.client = MagicMock()
    llm.client.chat.completions.create.side_effect = [
        _ModelLoadingError(),
        _make_response("ready now"),
    ]

    # patch out tenacity's sleep so the exponential backoff doesn't slow the test
    with patch("tenacity.nap.time.sleep"):
        result = llm.run_message("hello", use_thinking=False)

    assert result == "ready now"
    assert llm.client.chat.completions.create.call_count == 2


def test_run_message_returns_content() -> None:
    llm = LLM(model_url="http://localhost:1234", model_name="test-model")
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = _make_response("the answer")

    assert llm.run_message("hello", use_thinking=False) == "the answer"


def test_run_message_raises_on_no_output() -> None:
    llm = LLM(model_url="http://localhost:1234", model_name="test-model")
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = _make_response(None)

    with raises(NoOutputError):
        llm.run_message("hello", use_thinking=False)


@mark.parametrize(
    "use_thinking,expected_enabled",
    [
        param(True, True, id="thinking enabled"),
        param(False, False, id="thinking disabled"),
    ],
)
def test_run_message_passes_thinking_config(use_thinking, expected_enabled) -> None:
    llm = LLM(model_url="http://localhost:1234", model_name="test-model")
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = _make_response("ok")

    llm.run_message("hello", use_thinking=use_thinking)

    _, kwargs = llm.client.chat.completions.create.call_args
    extra_body = kwargs["extra_body"]
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is expected_enabled
    # timing reporting is always requested; harmless for servers that ignore it
    assert extra_body["timings_per_token"] is True
    # the model name is forwarded to the API
    assert kwargs["model"] == "test-model"


def test_log_tokens_per_second_silent_when_info_disabled(caplog) -> None:
    """Should be a no-op (and never raise) when info logging is off."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    response = SimpleNamespace(model_extra={}, usage=None)
    log_tokens_per_second(response, elapsed=1.0)
    assert caplog.records == []


def test_log_tokens_per_second_prefers_server_timings(caplog) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    response = SimpleNamespace(
        model_extra={
            "timings": {
                "predicted_per_second": 42.0,
                "predicted_n": 10,
                "predicted_ms": 250,
            }
        },
        usage=SimpleNamespace(completion_tokens=999),
    )
    log_tokens_per_second(response, elapsed=5.0)

    assert "throughput" in caplog.text
    # the server timing rate wins over the wall-clock fallback
    assert "42.0 tok/s" in caplog.text


def test_log_tokens_per_second_falls_back_to_usage(caplog) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    response = SimpleNamespace(
        model_extra={},
        usage=SimpleNamespace(completion_tokens=20),
    )
    log_tokens_per_second(response, elapsed=2.0)

    # 20 tokens over 2 seconds => 10 tok/s
    assert "10.0 tok/s" in caplog.text
