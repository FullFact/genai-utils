import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from genai_utils.llm import NoOutputError
from genai_utils.llm.fallback_model import FallbackChat, _describe
from genai_utils.llm.gemini import GeminiChat, ModelConfig
from genai_utils.llm.serverless_openai import ServerlessChat


def _fake_model(
    reply: str | None = None,
    error: BaseException | None = None,
    model_name: str | None = None,
) -> MagicMock:
    """Build a fake ``ChatModel`` that returns ``reply`` or raises ``error``."""
    model = MagicMock()
    model.model_name = model_name
    model.run_message = AsyncMock(return_value=reply, side_effect=error)
    model.aclose = AsyncMock()
    return model


async def test_uses_primary_when_it_succeeds() -> None:
    primary = _fake_model(reply="primary reply")
    fallback = _fake_model(reply="fallback reply")

    result = await FallbackChat(primary, fallback).run_message("hi", True)

    assert result == "primary reply"
    primary.run_message.assert_awaited_once_with("hi", True)
    fallback.run_message.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(NoOutputError(), id="no output"),
        pytest.param(asyncio.TimeoutError(), id="timeout"),
        pytest.param(ConnectionError("vast unreachable"), id="connection"),
        pytest.param(ValueError("unexpected"), id="other"),
    ],
)
async def test_falls_back_when_primary_raises(
    error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    primary = _fake_model(error=error)
    fallback = _fake_model(reply="fallback reply")

    result = await FallbackChat(primary, fallback).run_message("hi", False)

    assert result == "fallback reply"
    fallback.run_message.assert_awaited_once_with("hi", False)
    assert "using fallback model" in caplog.text


async def test_fallback_warning_has_fields_for_monitoring(
    caplog: pytest.LogCaptureFixture,
) -> None:
    primary = _fake_model(error=asyncio.TimeoutError(), model_name="claimants")
    fallback = _fake_model(reply="fallback reply")

    await FallbackChat(primary, fallback).run_message("hi", False)

    [record] = [r for r in caplog.records if r.name == "genai_utils.llm.fallback_model"]
    assert record.levelname == "WARNING"
    assert record.event_metric == "model_fallback"
    assert record.primary_model == "MagicMock(claimants)"
    assert record.fallback_model == "MagicMock"
    assert record.error_type == "TimeoutError"
    assert record.exc_info is not None


def test_describe_uses_model_name_when_the_client_has_one() -> None:
    serverless = ServerlessChat(endpoint_name="endpoint", model_name="claimants")
    gemini = GeminiChat(
        model_config=ModelConfig(
            project="project", location="location", model_name="gemini-3-flash"
        )
    )

    assert _describe(serverless) == "ServerlessChat(claimants)"
    assert _describe(gemini) == "GeminiChat"


async def test_fallback_error_is_raised() -> None:
    primary = _fake_model(error=ConnectionError("vast unreachable"))
    fallback = _fake_model(error=NoOutputError())

    with pytest.raises(NoOutputError):
        await FallbackChat(primary, fallback).run_message("hi", False)


async def test_cancellation_does_not_fall_back() -> None:
    primary = _fake_model(error=asyncio.CancelledError())
    fallback = _fake_model(reply="fallback reply")

    with pytest.raises(asyncio.CancelledError):
        await FallbackChat(primary, fallback).run_message("hi", False)
    fallback.run_message.assert_not_called()


async def test_aclose_closes_both_models() -> None:
    primary = _fake_model()
    fallback = _fake_model()

    await FallbackChat(primary, fallback).aclose()

    primary.aclose.assert_awaited_once()
    fallback.aclose.assert_awaited_once()


async def test_aclose_closes_fallback_when_primary_close_fails() -> None:
    primary = _fake_model()
    primary.aclose.side_effect = RuntimeError("close failed")
    fallback = _fake_model()

    with pytest.raises(RuntimeError, match="close failed"):
        await FallbackChat(primary, fallback).aclose()
    fallback.aclose.assert_awaited_once()
