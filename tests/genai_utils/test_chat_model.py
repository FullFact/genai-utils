import inspect

from pytest import mark, param

from genai_utils.llm.fallback_model import FallbackChat
from genai_utils.llm.gemini import GeminiChat, ModelConfig
from genai_utils.llm.models import ChatModel
from genai_utils.llm.openai import OpenaiChat
from genai_utils.llm.serverless_openai import ServerlessChat

_GEMINI_CONFIG = ModelConfig(project="p", location="l", model_name="m")


@mark.parametrize(
    "client",
    [
        param(OpenaiChat(model_url="http://localhost:1234"), id="OpenaiChat"),
        param(ServerlessChat(endpoint_name="ep"), id="ServerlessChat"),
        param(GeminiChat(model_config=_GEMINI_CONFIG), id="GeminiChat"),
        param(
            FallbackChat(
                primary=ServerlessChat(endpoint_name="ep"),
                fallback=GeminiChat(model_config=_GEMINI_CONFIG),
            ),
            id="FallbackChat",
        ),
    ],
)
def test_clients_implement_chat_model(client: ChatModel) -> None:
    assert isinstance(client, ChatModel)
    # isinstance only checks the methods exist, so check they're async too
    assert inspect.iscoroutinefunction(client.run_message)
    assert inspect.iscoroutinefunction(client.aclose)


def _accepts_chat_model(model: ChatModel) -> None:
    pass


def test_clients_type_check_as_chat_model() -> None:
    """Checked by mypy: every client is accepted where a ChatModel is."""
    _accepts_chat_model(OpenaiChat(model_url="http://localhost:1234"))
    _accepts_chat_model(ServerlessChat(endpoint_name="ep"))
    _accepts_chat_model(GeminiChat(model_config=_GEMINI_CONFIG))
    _accepts_chat_model(
        FallbackChat(
            primary=ServerlessChat(endpoint_name="ep"),
            fallback=GeminiChat(model_config=_GEMINI_CONFIG),
        )
    )
