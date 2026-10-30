import logging

from genai_utils.llm.models import ChatModel

_logger = logging.getLogger(__name__)

# Lets log-based metrics match the fallback warning without relying on its text.
FALLBACK_EVENT_METRIC = "model_fallback"


def _describe(model: ChatModel) -> str:
    """
    Names a model for logs, e.g. ``ServerlessChat(claimants)``.
    ``ChatModel`` has no name, so this uses ``model_name`` where the client has one.
    """
    model_name = getattr(model, "model_name", None)
    if model_name:
        return f"{type(model).__name__}({model_name})"
    return type(model).__name__


class FallbackChat(ChatModel):
    """
    A chat model with a primary and fallback model.
    Messages will be sent to the primary model first.
    If that fails, it will be sent to the fallback model.
    """

    primary: ChatModel
    fallback: ChatModel

    def __init__(self, primary: ChatModel, fallback: ChatModel):
        self.primary = primary
        self.fallback = fallback

    async def run_message(self, message_content: str, use_thinking: bool) -> str:
        """
        Sends the message to the primary model.
        If that fails, try the fallback model.
        Errors from the fallback model are raised to the caller.

        The fallback warning carries ``event_metric``, ``primary_model``,
        ``fallback_model`` and ``error_type`` as log record attributes, which
        structured log handlers emit as fields.
        """
        try:
            return await self.primary.run_message(message_content, use_thinking)
        except Exception as exc:
            _logger.warning(
                "Primary model failed; using fallback model.",
                exc_info=True,
                extra={
                    "event_metric": FALLBACK_EVENT_METRIC,
                    "primary_model": _describe(self.primary),
                    "fallback_model": _describe(self.fallback),
                    "error_type": type(exc).__name__,
                },
            )
            return await self.fallback.run_message(message_content, use_thinking)

    async def aclose(self) -> None:
        try:
            await self.primary.aclose()
        finally:
            await self.fallback.aclose()
