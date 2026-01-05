import os
from unittest.mock import Mock, patch

import pytest
from google.genai import Client
from google.genai.client import AsyncClient
from google.genai.models import Models

from genai_utils.gemini import (
    GeminiError,
    ModelConfig,
    run_prompt_async,
    validate_labels,
)


class DummyResponse:
    candidates = "yes!"
    text = "response!"


async def get_dummy():
    return DummyResponse()


def test_validate_labels_valid():
    """Test that valid labels pass validation"""
    valid_labels = {
        "team": "ai",
        "project": "genai-utils",
        "environment": "production",
        "version": "1-2-3",
        "my_label": "my_value",
    }
    # Should not raise any exception
    validate_labels(valid_labels)


def test_validate_labels_empty_key():
    """Test that empty keys are rejected"""
    with pytest.raises(GeminiError, match="cannot be empty"):
        validate_labels({"": "value"})


def test_validate_labels_key_too_long():
    """Test that keys exceeding 63 characters are rejected"""
    long_key = "a" * 64
    with pytest.raises(GeminiError, match="exceeds 63 characters"):
        validate_labels({long_key: "value"})


def test_validate_labels_value_too_long():
    """Test that values exceeding 63 characters are rejected"""
    long_value = "a" * 64
    with pytest.raises(GeminiError, match="exceeds 63 characters"):
        validate_labels({"key": long_value})


def test_validate_labels_key_starts_with_number():
    """Test that keys starting with numbers are rejected"""
    with pytest.raises(GeminiError, match="must start with a lowercase letter"):
        validate_labels({"1key": "value"})


def test_validate_labels_key_starts_with_uppercase():
    """Test that keys starting with uppercase are rejected"""
    with pytest.raises(GeminiError, match="must start with a lowercase letter"):
        validate_labels({"Key": "value"})


@pytest.mark.parametrize(
    "invalid_key", ["key@value", "key.name", "key$", "key with space", "key/name"]
)
def test_validate_labels_key_invalid_characters(invalid_key):
    """Test that keys with invalid characters are rejected"""
    with pytest.raises(GeminiError, match="contains invalid characters"):
        validate_labels({invalid_key: "value"})


@pytest.mark.parametrize(
    "invalid_value", ["value@", "value.txt", "value$", "value with space", "value/"]
)
def test_validate_labels_value_invalid_characters(invalid_value):
    """Test that values with invalid characters are rejected"""
    with pytest.raises(GeminiError, match="contains invalid characters"):
        validate_labels({"key": invalid_value})


def test_validate_labels_max_length_valid():
    """Test that keys and values at exactly 63 characters are valid"""
    max_key = "a" * 63
    max_value = "b" * 63
    # Should not raise any exception
    validate_labels({max_key: max_value})


def test_validate_labels_valid_special_chars():
    """Test that valid special characters (hyphens, underscores) are accepted"""
    valid_labels = {
        "my-key": "my-value",
        "my_key": "my_value",
        "my-key_name": "my-value_123",
        "key123": "value456",
    }
    # Should not raise any exception
    validate_labels(valid_labels)


@patch("genai_utils.gemini.genai.Client")
async def test_run_prompt_with_valid_labels(mock_client):
    """Test that run_prompt accepts and uses valid labels"""
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    labels = {"team": "ai", "project": "test"}

    await run_prompt_async(
        "test prompt",
        labels=labels,
        model_config=ModelConfig(
            project="project", location="location", model_name="model"
        ),
    )

    # Verify the call was made
    assert models.generate_content.called
    call_kwargs = models.generate_content.call_args[1]
    assert "config" in call_kwargs
    assert call_kwargs["config"].labels == labels


@patch("genai_utils.gemini.genai.Client")
async def test_run_prompt_with_invalid_labels(mock_client):
    """Test that run_prompt rejects invalid labels"""
    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    invalid_labels = {"Invalid": "value"}  # uppercase key

    with pytest.raises(GeminiError, match="must start with a lowercase letter"):
        await run_prompt_async(
            "test prompt",
            labels=invalid_labels,
            model_config=ModelConfig(
                project="project", location="location", model_name="model"
            ),
        )


@patch("genai_utils.gemini.genai.Client")
@patch.dict(os.environ, {"GENAI_LABEL_TEAM": "ai", "GENAI_LABEL_ENV": "test"})
async def test_run_prompt_merges_env_labels(mock_client):
    """Test that run_prompt merges environment labels with request labels"""
    # Need to reload the module to pick up the new environment variables
    import importlib

    import genai_utils.gemini

    importlib.reload(genai_utils.gemini)

    client = Mock(Client)
    models = Mock(Models)
    async_client = Mock(AsyncClient)

    models.generate_content.return_value = get_dummy()
    client.aio = async_client
    async_client.models = models
    mock_client.return_value = client

    request_labels = {"project": "test"}

    await genai_utils.gemini.run_prompt_async(
        "test prompt",
        labels=request_labels,
        model_config=ModelConfig(
            project="project", location="location", model_name="model"
        ),
    )

    # Verify the call was made with merged labels
    assert models.generate_content.called
    call_kwargs = models.generate_content.call_args[1]
    assert "config" in call_kwargs

    # Should contain both env labels (team, env) and request label (project)
    merged_labels = call_kwargs["config"].labels
    assert "team" in merged_labels
    assert "env" in merged_labels
    assert "project" in merged_labels
