"""
The interface shared by the LLM clients, so downstream code can switch
between them (e.g. a self-hosted ``openai.OpenaiChat`` and a vast.ai
``serverless_openai.ServerlessChat``) by changing only where the client is built.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatModel(Protocol):
    """
    An async client that sends a single user message to a model and returns
    the text of its reply.

    Constructor arguments differ between implementations (a URL vs a vast
    endpoint name, generation settings, ...), so they are deliberately not part
    of this protocol: depend on ``ChatModel`` at the call site and choose the
    implementation where it is constructed.

    Retries aren't part of this protocol either: clients that support them
    implement :class:`genai_utils.llm.retry.Retryable`.
    """

    async def run_message(self, message_content: str, use_thinking: bool) -> str:
        """
        Sends the message to the model and returns its reply.

        Parameters
        ----------
        message_content: str
            The message to send, as a single user turn.
        use_thinking: bool
            Whether the model should think before answering.

        Returns
        -------
        str
            The text of the model's reply.

        Raises
        ------
        NoOutputError:
            if the model returns no content.
        """
        ...

    async def aclose(self) -> None:
        """
        Releases the client's connections.
        Safe to call more than once.
        """
        ...


def describe_model(model: object) -> str:
    """
    Names a model for logs, e.g. ``ServerlessChat(claimants)``.
    ``ChatModel`` has no name, so this uses ``model_name`` where the client has one.
    """
    model_name = getattr(model, "model_name", None)
    if model_name:
        return f"{type(model).__name__}({model_name})"
    return type(model).__name__
