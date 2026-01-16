import asyncio
import logging
import os
import re
from typing import Any

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

_logger = logging.getLogger(__name__)

USER_AGENT = "FullFact (FullFact Media Ingest. https://fullfact.ai)"

DEFAULT_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

DEFAULT_PARAMETERS = {
    "candidate_count": 1,
    "temperature": 0,
    "top_p": 1,
}

DEFAULT_LABELS = {
    key.removeprefix("GENAI_LABEL_").lower(): value
    for key, value in os.environ.items()
    if key.startswith("GENAI_LABEL_")
}


class GeminiError(Exception):
    """
    Exception raised when something goes wrong with Gemini.
    """


class ModelConfig(BaseModel):
    """
    Config for a Gemini model.

    Attributes:
    -----------
    project: str
        The name of the project to run the model in.
    location: str
        The Google Cloud location at which to run the model.
        Different locations are capable of running different models.
        See `location info`_ in docs.
    model_name: str
        The name of the model you want to use.
        For example: `gemini-2.0-flash-lite`.

    .. _location info: https://cloud.google.com/vertex-ai/docs/general/locations
    """

    project: str
    location: str
    model_name: str


def generate_model_config() -> ModelConfig:
    """
    Generates a new model config from environment variables:
    `GEMINI_PROJECT`, `GEMINI_LOCATION`, and `GEMINI_MODEL`.
    """
    try:
        return ModelConfig(
            project=os.environ["GEMINI_PROJECT"],
            location=os.environ["GEMINI_LOCATION"],
            model_name=os.environ["GEMINI_MODEL"],
        )
    except KeyError as exc:
        message = (
            "You need to set the following environment variables: "
            + "GEMINI_PROJECT, "
            + "GEMINI_LOCATION, "
            + "GEMINI_MODEL"
        )
        raise GeminiError(message) from exc


def follow_redirect(uri: str) -> str:
    """
    Tries to follow a redirect to get the original url.
    """
    try:
        r = requests.get(uri, headers={"User-Agent": USER_AGENT}, timeout=10)
        og_uri = r.url or uri
        return og_uri
    except requests.exceptions.HTTPError as exc:
        _logger.warning(
            f"The link ({uri}) could not be followed. "
            "Falling back to vertex AI cached link."
            f"Details: {repr(exc)}"
        )
    except Exception as exc:
        _logger.warning(
            f"Something went wrong with ({uri}). "
            "Falling back to vertex AI cached link. "
            f"Details: {repr(exc)}"
        )
    return uri


def insert_citation(text: str, citation: str, citation_idx: int) -> str:
    """
    Finds the nearest appropriate place to put a citation.
    Appropriate here means in whitespace, preferably at the end of a sentence.
    Google's citation indices are not always accurate, so we need this
    to ensure that citations aren't inserted mid word.

    Args:
        text: The text to insert a citation into.
        citation: The citation text, e.g. '([1]("url"))'
        citation_idx: roughly where the citation should be inserted.

    Returns:
        The text with the citation inserted at the nearest appropriate location.
    """
    # first attempt to place a link at the end of a sentence (newline or sentence separator)
    places_for_link = [
        m.span()[0]
        for m in re.finditer(r"[\.\!\?\n][\s$]|(?<=[^0-9])[\.\!\?](?=[^0-9]|$)|$", text)
    ]
    # if that doesn't work, find the nearest white space
    if not places_for_link or not re.findall(r"[\.\!\?\n]", text):
        places_for_link = [m.span()[0] for m in re.finditer(r"\s", text)]
    # if that doesn't work, stick it on the end.
    if not places_for_link:
        return text + " " + citation
    distances = [citation_idx - p for p in places_for_link]
    nearest = min(distances, key=abs)
    targ_idx = citation_idx - nearest

    if targ_idx < len(text):
        output = text[:targ_idx] + " " + citation + text[targ_idx:]
    else:
        output = text + " " + citation
    return output


def add_citations(response: types.GenerateContentResponse) -> str:
    """
    Add citations to Gemini response text.
    """
    if not response.candidates or not response.text:
        raise GeminiError(
            f"No model output: possible reason: {response.prompt_feedback}"
        )
    text: str = response.text
    candidate: types.Candidate = response.candidates[0]
    if not candidate.grounding_metadata:
        _logger.info("No grounding metadata. Returning original text.")
        return text
    grounding_metadata: types.GroundingMetadata = candidate.grounding_metadata

    if not (
        grounding_metadata.grounding_supports and grounding_metadata.grounding_chunks
    ):
        _logger.info("No grounding supports or chunks. Returning original text.")
        return text

    supports = grounding_metadata.grounding_supports
    chunks = grounding_metadata.grounding_chunks

    links = []
    for chunk in chunks:
        if chunk.web and chunk.web.uri:
            url = chunk.web.uri
            links.append(follow_redirect(url))

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports: list[types.GroundingSupport] = sorted(
        supports,
        key=lambda s: s.segment.end_index,  # type: ignore
        reverse=True,
    )

    for support in sorted_supports:
        segment = support.segment
        if not isinstance(segment, types.Segment):
            continue
        end_index = segment.end_index
        if support.grounding_chunk_indices and isinstance(end_index, int):
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = links[i]
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            citation_string = f"({citation_string})"
            text = insert_citation(text, citation_string, end_index)

    return text


def validate_labels(labels: dict[str, str]) -> None:
    """
    Validates labels for GCP requirements.

    GCP label requirements:
    - Keys must start with a lowercase letter
    - Keys and values can only contain lowercase letters, numbers, hyphens, and underscores
    - Keys and values must be max 63 characters
    - Keys cannot be empty

    Raises:
        GeminiError: If labels don't meet GCP requirements
    """
    label_pattern = re.compile(r"^[a-z0-9_-]{1,63}$")
    key_start_pattern = re.compile(r"^[a-z]")

    for key, value in labels.items():
        if not key:
            raise GeminiError("Label keys cannot be empty")

        if len(key) > 63:
            raise GeminiError(
                f"Label key '{key}' exceeds 63 characters (length: {len(key)})"
            )

        if len(value) > 63:
            raise GeminiError(
                f"Label value for key '{key}' exceeds 63 characters (length: {len(value)})"
            )

        if not key_start_pattern.match(key):
            raise GeminiError(f"Label key '{key}' must start with a lowercase letter")

        if not label_pattern.match(key):
            raise GeminiError(
                f"Label key '{key}' contains invalid characters. "
                "Only lowercase letters, numbers, hyphens, and underscores are allowed"
            )

        if not label_pattern.match(value):
            raise GeminiError(
                f"Label value '{value}' for key '{key}' contains invalid characters. "
                "Only lowercase letters, numbers, hyphens, and underscores are allowed"
            )


def check_grounding_ran(response: types.GenerateContentResponse) -> bool:
    """
    Checks if grounding ran and logs some metadata about the grounding.
    """
    if not response.candidates:
        _logger.info("Grounding info: No candidates found, so grounding did not run.")
        return False
    grounding_metadata = response.candidates[0].grounding_metadata

    if not grounding_metadata:
        _logger.info("Grounding info: No grounding metadata was found.")
        return False

    n_searches = (
        len(grounding_metadata.web_search_queries)
        if grounding_metadata.web_search_queries
        else 0
    )
    n_chunks = (
        len(grounding_metadata.grounding_chunks)
        if grounding_metadata.grounding_chunks
        else 0
    )
    n_supports = (
        len(grounding_metadata.grounding_supports)
        if grounding_metadata.grounding_supports
        else 0
    )
    _logger.info(
        f"Grounding info: {n_searches} searches | {n_chunks} chunks | {n_supports} supports"
    )
    return bool(n_searches and n_chunks and n_supports)


def get_thinking_config(
    model_name: str, do_thinking: bool
) -> types.ThinkingConfig | None:
    """
    Gets the thinking cofig required for the current model.
    Thinking is set differently before and after Gemini 3.0.
    Certain models like the 2.5 and 3.0 pro models, do not allow grounding to be disabled.
    """
    if "gemini-2.5-pro" in model_name:
        if not do_thinking:
            _logger.warning(
                "It is not possible to turn off thinking with this model. Setting to minimum."
            )
            return types.ThinkingConfig(thinking_budget=128)  # minimum thinking
        return types.ThinkingConfig(thinking_budget=-1)  # dynamic budget

    if (
        model_name < "gemini-2.6"
    ):  # there is no 2.6, but this means it will catch all 2.5 variants
        if do_thinking:
            return types.ThinkingConfig(thinking_budget=-1)  # dynamic budget
        return types.ThinkingConfig(thinking_budget=0)  # disable thinking

    if model_name >= "gemini-3":
        if not do_thinking:
            if "pro" in model_name:
                _logger.warning(
                    "Cannot disable thinking in this model. Setting thinking to low."
                )
                return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
            return types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
        return None

    _logger.warning("Did not recognise the model provided, defaulting to None")
    return None


def run_prompt(
    prompt: str,
    video_uri: str | None = None,
    output_schema: types.SchemaUnion | None = None,
    system_instruction: str | None = None,
    generation_config: dict[str, Any] = DEFAULT_PARAMETERS,
    safety_settings: list[types.SafetySetting] = DEFAULT_SAFETY_SETTINGS,
    model_config: ModelConfig | None = None,
    use_grounding: bool = False,
    inline_citations: bool = False,
    labels: dict[str, str] = {},
) -> str:
    """
    A synchronous version of `run_prompt_async`.

    Parameters
    ----------
    prompt: str
        The prompt given to the model
    video_uri: str | None
        A Google Cloud URI for a video that you want to prompt.
    output_schema: types.SchemaUnion | None
        A valid schema for the model output.
        Generally, we'd recommend this being a pydantic BaseModel inheriting class,
        which defines the desired schema of the model output.
        ```python
        from pydantic import BaseModel, Field

        class Movie(BaseModel):
            title: str = Field(description="The title of the movie")
            year: int = Field(description="The year the film was released in the UK")

        schema = Movie
        # or
        schema = list[Movie]
        ```
        Use this if you want structured JSON output.
    system_instruction: str | None
        An instruction to the model which essentially goes before the prompt.
        For example:
        ```
        You are a fact checker and you must base all your answers on evidence
        ```
    generation_config: dict[str, Any]
        The parameters for the generation. See the docs (`generation config`_).
    safety_settings: dict[generative_models.HarmCategory, generative_models.HarmBlockThreshold]
        The safety settings for generation. Determines what will be blocked.
        See the docs (`safety settings`_)
    model_config: ModelConfig | None
        The config for the Gemini model.
        Specifies project, location, and model name.
        If None, will attempt to use environment variables:
        `GEMINI_PROJECT`, `GEMINI_LOCATION`, and `GEMINI_MODEL`.
    use_grounding: bool
        Whether Gemini should perform a Google search to ground results.
        This will allow it to pull from up-to-date information,
        and makes the output more likely to be factual.
        Does not work with structured output.
        See the docs (`grounding`_).
    do_thinking: bool
        Whether Gemini should use a thought process.
        This is more expensive but may yield better results.
        Do not use for bulk tasks that don't require complex thoughts.
    inline_citations: bool
        Whether output should include citations inline with the text.
        These citations will be links to be used as evidence.
        This is only possible if grounding is set to true.
    labels: dict[str, str]
        Optional labels to attach to the API call for tracking and monitoring purposes.
        Labels are key-value pairs that can be used to organize and filter requests
        in Google Cloud logs and metrics.

    Returns
    -------
    The text output of the Gemini model.

    .. _generation config: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig
    .. _safety settings: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters
    .. _grounding: https://ai.google.dev/gemini-api/docs/google-search
    """
    return asyncio.run(
        run_prompt_async(
            prompt=prompt,
            video_uri=video_uri,
            output_schema=output_schema,
            system_instruction=system_instruction,
            generation_config=generation_config,
            safety_settings=safety_settings,
            model_config=model_config,
            use_grounding=use_grounding,
            inline_citations=inline_citations,
            labels=labels,
        )
    )


async def run_prompt_async(
    prompt: str,
    video_uri: str | None = None,
    output_schema: types.SchemaUnion | None = None,
    system_instruction: str | None = None,
    generation_config: dict[str, Any] = DEFAULT_PARAMETERS,
    safety_settings: list[types.SafetySetting] = DEFAULT_SAFETY_SETTINGS,
    model_config: ModelConfig | None = None,
    use_grounding: bool = False,
    do_thinking: bool = False,
    inline_citations: bool = False,
    labels: dict[str, str] = {},
) -> str:
    """
    Runs a prompt through the model.

    Parameters
    ----------
    prompt: str
        The prompt given to the model
    video_uri: str | None
        A Google Cloud URI for a video that you want to prompt.
    output_schema: types.SchemaUnion | None
        A valid schema for the model output.
        Generally, we'd recommend this being a pydantic BaseModel inheriting class,
        which defines the desired schema of the model output.
        ```python
        from pydantic import BaseModel, Field

        class Movie(BaseModel):
            title: str = Field(description="The title of the movie")
            year: int = Field(description="The year the film was released in the UK")

        schema = Movie
        # or
        schema = list[Movie]
        ```
        Use this if you want structured JSON output.
    system_instruction: str | None
        An instruction to the model which essentially goes before the prompt.
        For example:
        ```
        You are a fact checker and you must base all your answers on evidence
        ```
    generation_config: dict[str, Any]
        The parameters for the generation. See the docs (`generation config`_).
    safety_settings: dict[generative_models.HarmCategory, generative_models.HarmBlockThreshold]
        The safety settings for generation. Determines what will be blocked.
        See the docs (`safety settings`_)
    model_config: ModelConfig | None
        The config for the Gemini model.
        Specifies project, location, and model name.
        If None, will attempt to use environment variables:
        `GEMINI_PROJECT`, `GEMINI_LOCATION`, and `GEMINI_MODEL`.
    use_grounding: bool
        Whether Gemini should perform a Google search to ground results.
        This will allow it to pull from up-to-date information,
        and makes the output more likely to be factual.
        Does not work with structured output.
        See the docs (`grounding`_).
    do_thinking: bool
        Whether Gemini should use a thought process.
        This is more expensive but may yield better results.
        Do not use for bulk tasks that don't require complex thoughts.
    inline_citations: bool
        Whether output should include citations inline with the text.
        These citations will be links to be used as evidence.
        This is only possible if grounding is set to true.
    labels: dict[str, str]
        Optional labels to attach to the API call for tracking and monitoring purposes.
        Labels are key-value pairs that can be used to organize and filter requests
        in Google Cloud logs and metrics.

    Returns
    -------
    The text output of the Gemini model.

    .. _generation config: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig
    .. _safety settings: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters
    .. _grounding: https://ai.google.dev/gemini-api/docs/google-search
    """
    # make a copy of the generation config so it doesn't change between runs
    built_gen_config = {**generation_config}
    if model_config is None:
        model_config = generate_model_config()

    client = genai.Client(
        vertexai=True,
        project=model_config.project,
        location=model_config.location,
    )

    # construct the input, adding the video if provided
    parts = []
    if video_uri:
        parts.append(types.Part.from_uri(file_uri=video_uri, mime_type="video/mp4"))

    parts.append(types.Part.from_text(text=prompt))

    # define the schema for the output of the model
    if output_schema:
        built_gen_config["response_mime_type"] = "application/json"
        built_gen_config["response_schema"] = output_schema

    # sort out grounding if required
    if use_grounding:
        if output_schema:
            raise GeminiError(
                "You cannot use structured output and grounding together."
            )
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        built_gen_config["tools"] = [grounding_tool]

    if inline_citations and not use_grounding:
        raise GeminiError("Inline citations only work if `use_grounding = True`")
    merged_labels = DEFAULT_LABELS | labels
    validate_labels(merged_labels)

    response = await client.aio.models.generate_content(
        model=model_config.model_name,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            **built_gen_config,
            labels=merged_labels,
            thinking_config=get_thinking_config(model_config.model_name, do_thinking),
        ),
    )

    if use_grounding:
        grounding_ran = check_grounding_ran(response)
        if not grounding_ran:
            _logger.warning(
                "Grounding Info: GROUNDING FAILED - see previous log messages for reason"
            )

    if response.candidates and response.text and isinstance(response.text, str):
        if inline_citations and response.candidates[0].grounding_metadata:
            text_with_citations = add_citations(response)
            return text_with_citations
        else:
            return response.text

    raise GeminiError(f"No model output: possible reason: {response.prompt_feedback}")
