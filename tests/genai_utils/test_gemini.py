import os
from unittest.mock import Mock, patch

from google.genai import Client
from google.genai.models import Models
from pydantic import BaseModel, Field

from genai_utils.gemini import (
    DEFAULT_PARAMETERS,
    GeminiError,
    ModelConfig,
    generate_model_config,
    run_prompt,
)


class DummyResponse:
    candidates = "yes!"
    text = "response!"


class DummySchema(BaseModel):
    breed: str = Field(description="Breed of dog")
    colour: str = Field(description="Colour of dog")


def get_dummy():
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
def test_dont_overwrite_generation_config(mock_client):
    copy_of_params = {**DEFAULT_PARAMETERS}
    client = Mock(Client)
    models = Mock(Models)

    models.generate_content.return_value = get_dummy()
    client.models = models
    mock_client.return_value = client

    assert DEFAULT_PARAMETERS == copy_of_params
    run_prompt(
        "do something",
        output_schema=DummySchema,
        model_config=ModelConfig(
            project="project", location="location", model_name="model"
        ),
    )
    models.generate_content.return_value = get_dummy()
    run_prompt(
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
def test_error_if_grounding_with_schema(mock_client):
    client = Mock(Client)
    models = Mock(Models)

    models.generate_content.return_value = get_dummy()
    client.models = models
    mock_client.return_value = client

    try:
        run_prompt(
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
def test_error_if_citations_and_no_grounding(mock_client):
    client = Mock(Client)
    models = Mock(Models)

    models.generate_content.return_value = get_dummy()
    client.models = models
    mock_client.return_value = client

    try:
        run_prompt(
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
