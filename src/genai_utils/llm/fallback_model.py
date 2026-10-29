import logging

from genai_utils.llm.models import ChatModel

_logger = logging.getLogger(__name__)


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
        """
        try:
            return await self.primary.run_message(message_content, use_thinking)
        except Exception:
            _logger.warning(
                "Primary model failed; using fallback model.", exc_info=True
            )
            return await self.fallback.run_message(message_content, use_thinking)

    async def aclose(self) -> None:
        try:
            await self.primary.aclose()
        finally:
            await self.fallback.aclose()
