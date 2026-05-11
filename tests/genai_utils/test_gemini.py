import os
from unittest.mock import Mock, patch

from google.genai import Client, types
from google.genai.client import AsyncClient
from google.genai.models import Models
from pydantic import BaseModel, Field
from pytest import mark, param, raises

from genai_utils.gemini import (
    DEFAULT_PARAMETERS,
    GeminiError,
    ModelConfig,
    NoGroundingError,
    generate_model_config,
    get_thinking_config,
    run_prompt_async,
)


class DummyResponse:
    candidates = "yes!"
    text = "response!"


class DummySchema(BaseModel):
    breed: str = Field(description="Breed of dog")
    colour: str = Field(description="Colour of dog")


async def get_dummy():
    return DummyResponse()


def test_generate_model_config():
    os.environ["GEMINI_MODEL"] = "m"

    config = generate_model_config()
    assert config.model_name == "m"


def test_generate_model_config_no_env_vars():
    if "GEMINI_MODEL" in os.environ:
        os.environ.pop("GEMINI_MODEL")

    with raises(GeminiError):
        _ = generate_model_config()


@patch("genai_utils.gemini.genai.Client")
async def test_dont_overwrite_generation_config(mock_client):
    copy_of_params = {**DEFAULT_PARAMETERS}
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    assert DEFAULT_PARAMETERS == copy_of_params
    await run_prompt_async(
        "do something",
        output_schema=DummySchema,
        model_config=ModelConfig(model_name="model"),
    )
    models.generate_content.return_value = get_dummy()
    await run_prompt_async(
        "do something",
        model_config=ModelConfig(model_name="model"),
    )
    assert DEFAULT_PARAMETERS == copy_of_params

    call_args = models.generate_content.call_args_list
    assert call_args[0][1]["config"].response_mime_type == "application/json"
    assert call_args[1][1]["config"].response_mime_type is None


@patch("genai_utils.gemini.genai.Client")
async def test_error_if_grounding_with_schema(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    with raises(GeminiError):
        await run_prompt_async(
            "do something",
            output_schema=DummySchema,
            use_grounding=True,
            model_config=ModelConfig(model_name="model"),
        )


@patch("genai_utils.gemini.genai.Client")
async def test_error_if_citations_and_no_grounding(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    with raises(GeminiError):
        await run_prompt_async(
            "do something",
            use_grounding=False,
            inline_citations=True,
            model_config=ModelConfig(model_name="model"),
        )


@patch("genai_utils.gemini.genai.Client")
async def test_no_grounding_error_when_grounding_does_not_run(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    async def get_no_grounding_metadata_response():
        candidate = Mock()
        candidate.grounding_metadata = None
        response = Mock()
        response.candidates = [candidate]
        response.text = "response!"
        return response

    models.generate_content.return_value = get_no_grounding_metadata_response()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    with raises(NoGroundingError):
        await run_prompt_async(
            "do something",
            use_grounding=True,
            model_config=ModelConfig(model_name="model"),
        )


@mark.parametrize(
    "model_name,do_thinking,expected",
    [
        param("gemini-2.0-flash", False, types.ThinkingConfig(thinking_budget=0)),
        param("gemini-2.0-flash", True, types.ThinkingConfig(thinking_budget=-1)),
        param("gemini-2.5-flash-lite", False, types.ThinkingConfig(thinking_budget=0)),
        param("gemini-2.5-flash-lite", True, types.ThinkingConfig(thinking_budget=-1)),
        param("gemini-2.5-pro", False, types.ThinkingConfig(thinking_budget=128)),
        param("gemini-2.5-pro", True, types.ThinkingConfig(thinking_budget=-1)),
        param(
            "gemini-3.0-flash",
            False,
            types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        ),
        param("gemini-3.0-flash", True, None),
        param(
            "gemini-3.0-pro",
            False,
            types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
        param("gemini-3.0-pro", True, None),
    ],
)
def test_get_thinking_config(
    model_name: str, do_thinking: bool, expected: types.ThinkingConfig
):
    thinking_config = get_thinking_config(model_name, do_thinking)
    assert thinking_config == expected


# --- flex_pricing ---


@patch("genai_utils.gemini.genai.Client")
async def test_flex_pricing_sets_service_tier_and_timeout(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    await run_prompt_async(
        "do something",
        flex_pricing=True,
        model_config=ModelConfig(model_name="model"),
    )

    _, client_kwargs = mock_client.call_args
    assert client_kwargs["http_options"].timeout == 900_000

    call_kwargs = models.generate_content.call_args[1]
    assert call_kwargs["config"].service_tier == types.ServiceTier.FLEX


@patch("genai_utils.gemini.genai.Client")
async def test_no_flex_pricing_passes_no_http_options(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    await run_prompt_async(
        "do something",
        model_config=ModelConfig(model_name="model"),
    )

    _, client_kwargs = mock_client.call_args
    assert client_kwargs["http_options"] is None

    call_kwargs = models.generate_content.call_args[1]
    assert call_kwargs["config"].service_tier is None


# --- run_prompt_async happy path ---


@patch("genai_utils.gemini.genai.Client")
async def test_run_prompt_async_returns_text(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    response = Mock()
    response.candidates = ["yes!"]
    response.text = "response!"

    async def get_response():
        return response

    models.generate_content.return_value = get_response()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    result = await run_prompt_async(
        "do something",
        model_config=ModelConfig(model_name="gemini-2.0-flash"),
    )
    assert result == "response!"


@patch("genai_utils.gemini.genai.Client")
async def test_run_prompt_async_raises_when_no_output(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    response = Mock()
    response.candidates = None
    response.text = None
    response.prompt_feedback = "blocked"

    async def get_response():
        return response

    models.generate_content.return_value = get_response()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    with raises(GeminiError):
        await run_prompt_async(
            "do something",
            model_config=ModelConfig(model_name="model"),
        )


# --- video URI handling ---


@patch("genai_utils.gemini.genai.Client")
async def test_video_uri_youtube_passes_through(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    files = Mock()

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    async_client.files = files
    mock_client.return_value = client

    await run_prompt_async(
        "summarise",
        video_uri="https://www.youtube.com/watch?v=abc",
        model_config=ModelConfig(model_name="model"),
    )

    files.upload.assert_not_called()
    contents = models.generate_content.call_args[1]["contents"]
    video_part = contents.parts[0]
    assert video_part.file_data.file_uri == "https://www.youtube.com/watch?v=abc"


@patch("genai_utils.gemini.genai.Client")
async def test_video_uri_local_file_uploads(mock_client, tmp_path):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    files = Mock()

    uploaded = Mock()
    uploaded.uri = "files/abc"
    uploaded.mime_type = "video/mp4"
    uploaded.state = types.FileState.ACTIVE
    uploaded.name = "files/abc"

    async def upload(file):
        return uploaded

    files.upload = upload

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    async_client.files = files
    mock_client.return_value = client

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")

    await run_prompt_async(
        "summarise",
        video_uri=str(video_path),
        model_config=ModelConfig(model_name="model"),
    )

    contents = models.generate_content.call_args[1]["contents"]
    video_part = contents.parts[0]
    assert video_part.file_data.file_uri == "files/abc"


@patch("genai_utils.gemini.genai.Client")
async def test_video_uri_missing_local_file_raises(mock_client):
    client = Mock(Client)
    async_client = Mock(AsyncClient)
    client.aio = async_client
    mock_client.return_value = client

    with raises(GeminiError):
        await run_prompt_async(
            "summarise",
            video_uri="/nonexistent/path/video.mp4",
            model_config=ModelConfig(model_name="model"),
        )
