import os
from unittest.mock import Mock, patch

import requests as req
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
    add_citations,
    check_grounding_ran,
    follow_redirect,
    generate_model_config,
    get_thinking_config,
    insert_citation,
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


# --- follow_redirect ---


@patch("genai_utils.gemini.requests.get")
def test_follow_redirect_success(mock_get):
    mock_response = Mock()
    mock_response.url = "https://real-url.com"
    mock_get.return_value = mock_response

    result = follow_redirect("https://short.url")
    assert result == "https://real-url.com"


@mark.parametrize(
    "exception",
    [
        param(req.exceptions.HTTPError("error"), id="http-error"),
        param(Exception("something went wrong"), id="generic-exception"),
    ],
)
@patch("genai_utils.gemini.requests.get")
def test_follow_redirect_error_falls_back(mock_get, exception):
    mock_get.side_effect = exception
    assert follow_redirect("https://original.url") == "https://original.url"


# --- insert_citation ---


@mark.parametrize(
    "text,citation_idx,expected",
    [
        param("Hello world.", 12, "Hello world. (cite)", id="sentence-end"),
        param("Hello world", 5, "Hello (cite) world", id="whitespace-fallback"),
        param("Helloworld", 5, "Helloworld (cite)", id="no-whitespace-append"),
    ],
)
def test_insert_citation(text, citation_idx, expected):
    assert insert_citation(text, "(cite)", citation_idx) == expected


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


# --- check_grounding_ran ---


def test_check_grounding_ran_no_candidates():
    response = Mock()
    response.candidates = []
    assert check_grounding_ran(response) is False


def test_check_grounding_ran_no_grounding_metadata():
    candidate = Mock()
    candidate.grounding_metadata = None
    response = Mock()
    response.candidates = [candidate]
    assert check_grounding_ran(response) is False


def test_check_grounding_ran_returns_true_when_grounding_present():
    metadata = Mock()
    metadata.web_search_queries = ["query"]
    metadata.grounding_chunks = [Mock()]
    metadata.grounding_supports = [Mock()]
    candidate = Mock()
    candidate.grounding_metadata = metadata
    response = Mock()
    response.candidates = [candidate]
    assert check_grounding_ran(response) is True


def test_check_grounding_ran_returns_false_when_no_searches():
    metadata = Mock()
    metadata.web_search_queries = []
    metadata.grounding_chunks = [Mock()]
    metadata.grounding_supports = [Mock()]
    candidate = Mock()
    candidate.grounding_metadata = metadata
    response = Mock()
    response.candidates = [candidate]
    assert check_grounding_ran(response) is False


# --- add_citations ---


@mark.parametrize(
    "candidates,text",
    [
        param(None, None, id="no-candidates"),
        param([Mock()], None, id="no-text"),
    ],
)
def test_add_citations_raises_when_missing_output(candidates, text):
    response = Mock()
    response.candidates = candidates
    response.text = text
    response.prompt_feedback = "blocked"
    with raises(GeminiError):
        add_citations(response)


def test_add_citations_returns_plain_text_when_no_grounding_metadata():
    candidate = Mock()
    candidate.grounding_metadata = None
    response = Mock()
    response.candidates = [candidate]
    response.text = "plain text"
    assert add_citations(response) == "plain text"


def test_add_citations_returns_plain_text_when_no_supports():
    metadata = Mock()
    metadata.grounding_supports = None
    metadata.grounding_chunks = [Mock()]
    candidate = Mock()
    candidate.grounding_metadata = metadata
    response = Mock()
    response.candidates = [candidate]
    response.text = "plain text"
    assert add_citations(response) == "plain text"


def test_add_citations_returns_plain_text_when_no_chunks():
    metadata = Mock()
    metadata.grounding_supports = [Mock()]
    metadata.grounding_chunks = None
    candidate = Mock()
    candidate.grounding_metadata = metadata
    response = Mock()
    response.candidates = [candidate]
    response.text = "plain text"
    assert add_citations(response) == "plain text"


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
        model_config=ModelConfig(project="p", location="l", model_name="gemini-2.0-flash"),
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
            model_config=ModelConfig(project="p", location="l", model_name="model"),
        )
