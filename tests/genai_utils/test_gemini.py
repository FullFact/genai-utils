import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

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
    aclose_clients,
    close_clients,
    generate_model_config,
    get_async_client,
    get_thinking_config,
    run_prompt,
    run_prompt_async,
    validate_labels,
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
    os.environ["GEMINI_PROJECT"] = "p"
    os.environ["GEMINI_LOCATION"] = "l"
    os.environ["GEMINI_MODEL"] = "m"

    config = generate_model_config()
    assert config.project == "p"
    assert config.location == "l"
    assert config.model_name == "m"


def test_generate_model_config_no_env_vars():
    if "GEMINI_PROJECT" in os.environ:
        os.environ.pop("GEMINI_PROJECT")
    if "GEMINI_LOCATION" in os.environ:
        os.environ.pop("GEMINI_LOCATION")
    if "GEMINI_MODEL" in os.environ:
        os.environ.pop("GEMINI_MODEL")

    try:
        _ = generate_model_config()
    except GeminiError:
        assert True
        return

    assert False


@patch("genai_utils.gemini.genai.Client")
async def test_dont_overwrite_generation_config(mock_client):
    copy_of_params = {**DEFAULT_PARAMETERS}
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    async_client.aclose = AsyncMock()

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    assert DEFAULT_PARAMETERS == copy_of_params
    await run_prompt_async(
        "do something",
        output_schema=DummySchema,
        model_config=ModelConfig(
            project="project", location="location", model_name="model"
        ),
    )
    models.generate_content.return_value = get_dummy()
    await run_prompt_async(
        "do something",
        model_config=ModelConfig(
            project="project", location="location", model_name="model"
        ),
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
    async_client.aclose = AsyncMock()

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    try:
        await run_prompt_async(
            "do something",
            output_schema=DummySchema,
            use_grounding=True,
            model_config=ModelConfig(
                project="project", location="location", model_name="model"
            ),
        )
    except GeminiError:
        assert True
        return

    assert False


@patch("genai_utils.gemini.genai.Client")
async def test_error_if_citations_and_no_grounding(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    async_client.aclose = AsyncMock()

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    try:
        await run_prompt_async(
            "do something",
            use_grounding=False,
            inline_citations=True,
            model_config=ModelConfig(
                project="project", location="location", model_name="model"
            ),
        )
    except GeminiError:
        assert True
        return

    assert False


@patch("genai_utils.gemini.genai.Client")
async def test_no_grounding_error_when_grounding_does_not_run(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    async_client.aclose = AsyncMock()

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
            model_config=ModelConfig(
                project="project", location="location", model_name="model"
            ),
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


# --- validate_labels ---


def test_validate_labels_valid():
    labels = {"valid-key": "valid-value", "another_key": "value-123"}
    assert validate_labels(labels) == labels


@mark.parametrize(
    "labels",
    [
        param({"": "value"}, id="empty-key"),
        param({"a" * 64: "value"}, id="key-too-long"),
        param({"key": "a" * 64}, id="value-too-long"),
        param({"1key": "value"}, id="key-starts-with-digit"),
        param({"_key": "value"}, id="key-starts-with-underscore"),
        param({"KEY": "value"}, id="key-uppercase"),
        param({"key.dots": "value"}, id="key-with-dots"),
        param({"key": "VALUE"}, id="value-uppercase"),
        param({"key": "val ue"}, id="value-with-space"),
    ],
)
def test_validate_labels_invalid_input_dropped(labels):
    assert validate_labels(labels) == {}


def test_validate_labels_mixed_keeps_only_valid():
    labels = {"valid": "ok", "INVALID": "value", "": "empty"}
    assert validate_labels(labels) == {"valid": "ok"}


# --- run_prompt_async happy path ---


@patch("genai_utils.gemini.genai.Client")
async def test_run_prompt_async_returns_text(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    async_client.aclose = AsyncMock()

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
        model_config=ModelConfig(
            project="p", location="l", model_name="gemini-2.0-flash"
        ),
    )
    assert result == "response!"


@patch("genai_utils.gemini.genai.Client")
async def test_run_prompt_async_raises_when_no_output(mock_client):
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    async_client.aclose = AsyncMock()

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
            model_config=ModelConfig(project="p", location="l", model_name="model"),
        )


# --- client reuse (one client per project/location, not per prompt) ---


def _sync_mock_client():
    client = Mock(Client)
    models = Mock(Models)
    response = Mock()
    response.candidates = ["yes!"]
    response.text = "response!"
    models.generate_content.return_value = response
    client.models = models
    return client


@patch("genai_utils.gemini.genai.Client")
def test_run_prompt_reuses_one_client(mock_client):
    """Building a client resolves credentials and holds native memory, so the
    sync path must build one per project/location, not one per prompt."""
    mock_client.return_value = _sync_mock_client()
    config = ModelConfig(project="p", location="l", model_name="gemini-2.0-flash")

    assert run_prompt("do something", model_config=config) == "response!"
    assert run_prompt("do something else", model_config=config) == "response!"

    mock_client.assert_called_once()
    mock_client.return_value.close.assert_not_called()


@patch("genai_utils.gemini.genai.Client")
def test_run_prompt_builds_a_client_per_project_and_location(mock_client):
    mock_client.return_value = _sync_mock_client()

    run_prompt(
        "do something",
        model_config=ModelConfig(project="p", location="l", model_name="model"),
    )
    run_prompt(
        "do something",
        model_config=ModelConfig(project="p", location="other", model_name="model"),
    )
    # A different model on an existing project/location reuses its client.
    run_prompt(
        "do something",
        model_config=ModelConfig(project="p", location="l", model_name="other-model"),
    )

    assert mock_client.call_count == 2


@patch("genai_utils.gemini.genai.Client")
def test_close_clients_closes_and_forgets_them(mock_client):
    mock_client.return_value = _sync_mock_client()
    config = ModelConfig(project="p", location="l", model_name="model")

    run_prompt("do something", model_config=config)
    close_clients()

    mock_client.return_value.close.assert_called_once()
    # Forgotten as well as closed, so the next call builds a fresh one.
    run_prompt("do something", model_config=config)
    assert mock_client.call_count == 2


def _async_mock_client():
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)
    async_client.aclose = AsyncMock()

    response = Mock()
    response.candidates = ["yes!"]
    response.text = "response!"

    async def get_response():
        return response

    # A new coroutine per call, since a coroutine can only be awaited once.
    models.generate_content.side_effect = lambda **_: get_response()
    client.aio = async_client
    async_client.models = models
    return client


@patch("genai_utils.gemini.genai.Client")
async def test_run_prompt_async_reuses_one_client_per_loop(mock_client):
    mock_client.return_value = _async_mock_client()
    config = ModelConfig(project="p", location="l", model_name="model")

    assert await run_prompt_async("do something", model_config=config) == "response!"
    assert await run_prompt_async("and again", model_config=config) == "response!"

    mock_client.assert_called_once()
    # The transport stays open for the next prompt on this loop.
    mock_client.return_value.aio.aclose.assert_not_awaited()


@patch("genai_utils.gemini.genai.Client")
def test_async_clients_are_not_shared_between_loops(mock_client):
    """The async transport binds to the loop it is first used on, so a client
    cannot outlive its loop -- a caller using `asyncio.run` per batch gets one
    client per batch rather than one per prompt."""
    mock_client.return_value = _async_mock_client()
    config = ModelConfig(project="p", location="l", model_name="model")

    async def two_prompts():
        await run_prompt_async("one", model_config=config)
        await run_prompt_async("two", model_config=config)

    asyncio.run(two_prompts())
    asyncio.run(two_prompts())

    assert mock_client.call_count == 2


@patch("genai_utils.gemini.genai.Client")
async def test_aclose_clients_closes_this_loops_clients(mock_client):
    mock_client.return_value = _async_mock_client()
    config = ModelConfig(project="p", location="l", model_name="model")

    await run_prompt_async("do something", model_config=config)
    await aclose_clients()

    mock_client.return_value.aio.aclose.assert_awaited_once()
    await run_prompt_async("do something", model_config=config)
    assert mock_client.call_count == 2


@patch("genai_utils.gemini.genai.Client")
async def test_get_async_client_returns_the_same_client_within_a_loop(mock_client):
    mock_client.return_value = _async_mock_client()
    config = ModelConfig(project="p", location="l", model_name="model")

    assert get_async_client(config) is get_async_client(config)


def test_get_async_client_needs_a_running_loop():
    config = ModelConfig(project="p", location="l", model_name="model")

    with raises(RuntimeError):
        get_async_client(config)
