from unittest.mock import Mock, patch

import requests
from google.genai import types
from pytest import mark, param, raises

from genai_utils.gemini import (
    GeminiError,
    add_citations,
    check_grounding_ran,
    follow_redirect,
    insert_citation,
)

dummy_supports = [
    types.GroundingSupport(
        segment=types.Segment(end_index=30), grounding_chunk_indices=[0]
    ),
    types.GroundingSupport(
        segment=types.Segment(end_index=40), grounding_chunk_indices=[1, 2]
    ),
    types.GroundingSupport(
        segment=types.Segment(end_index=62), grounding_chunk_indices=[0, 2]
    ),
]
dummy_chunks = [
    types.GroundingChunk(web=types.GroundingChunkWeb(uri="first.com")),
    types.GroundingChunk(web=types.GroundingChunkWeb(uri="second.com")),
    types.GroundingChunk(web=types.GroundingChunkWeb(uri="third.com")),
]
dummy_grounding = types.GroundingMetadata(
    grounding_supports=dummy_supports,
    grounding_chunks=dummy_chunks,
    web_search_queries=["search 1", "search 2"],
)
dodgy_grounding = types.GroundingMetadata(
    grounding_supports=None, grounding_chunks=None
)


def dummy_redirect(uri: str) -> str:
    return uri + "/redirect"


@mark.parametrize(
    "text,grounding,expected",
    [
        param(
            ("this is the first sentence. and the second. and the third one"),
            dummy_grounding,
            (
                "this is the first sentence ([1](first.com/redirect)). "
                "and the second ([2](second.com/redirect), [3](third.com/redirect)). "
                "and the third one ([1](first.com/redirect), [3](third.com/redirect))"
            ),
            id="with normal grounding object",
        ),
        param(
            "this is a sentence",
            None,
            "this is a sentence",
            id="no grounding data",
        ),
        param(
            "this is a sentence",
            dodgy_grounding,
            "this is a sentence",
            id="with dodgy grounding object",
        ),
    ],
)
@patch("genai_utils.gemini.follow_redirect", wraps=dummy_redirect)
def test_add_citations(
    _mock_follow_redirect: Mock,
    text: str,
    grounding: types.GroundingMetadata | None,
    expected: str,
) -> None:
    response = Mock(types.GenerateContentResponse)
    response.candidates = [types.Candidate(grounding_metadata=grounding)]
    response.text = text
    result = add_citations(response)
    assert result == expected


@mark.parametrize(
    "og_url,raises,expected",
    [
        param("example.com", None, "redirected.com", id="normal case"),
        param(
            "example.com",
            requests.exceptions.HTTPError(),
            "example.com",
            id="HTTP error",
        ),
        param("example.com", Exception(), "example.com", id="Other error"),
        param(
            "example.com",
            requests.exceptions.Timeout,
            "example.com",
            id="Timeout error",
        ),
    ],
)
@patch("genai_utils.gemini.requests.get")
def test_follow_redirect(
    mock_get: Mock, og_url: str, raises: BaseException | None, expected: str
) -> None:
    class FakeRequest:
        url: str = "redirected.com"

    if raises:
        mock_get.side_effect = raises
    else:
        mock_get.return_value = FakeRequest()
    redirected = follow_redirect(og_url)
    assert redirected == expected


@mark.parametrize(
    "text,citation,citation_idx,expected",
    [
        param(
            (
                "This is a sentence. "
                "And another one. "
                "There are 2.5 things wrong with it."
                "Grand finale."
            ),
            "[desc](url)",
            4,
            (
                "This is a sentence [desc](url). "
                "And another one. "
                "There are 2.5 things wrong with it."
                "Grand finale."
            ),
            id="normal example",
        ),
        param(
            "one.two.three.four.",
            "[cit]",
            3,
            "one [cit].two.three.four.",
            id="no whitespace",
        ),
        param(
            "one two three four",
            "[cit]",
            4,
            "one [cit] two three four",
            id="No punctuation",
        ),
        param(
            "one\n two three four",
            "[cit]",
            4,
            "one [cit]\n two three four",
            id="No punctuation but newline",
        ),
        param(
            "onetwothreefour",
            "[cit]",
            4,
            "onetwothreefour [cit]",
            id="no whitespace or punctuation",
        ),
    ],
)
def test_insert_citation(
    text: str, citation: str, citation_idx: int, expected: str
) -> None:
    inserted = insert_citation(text, citation, citation_idx)
    assert inserted == expected


@mark.parametrize(
    "response, expected",
    [
        param(types.GenerateContentResponse(candidates=None), False),
        param(
            types.GenerateContentResponse(
                candidates=[types.Candidate(grounding_metadata=None)]
            ),
            False,
        ),
        param(
            types.GenerateContentResponse(
                candidates=[types.Candidate(grounding_metadata=dummy_grounding)]
            ),
            True,
        ),
        param(
            types.GenerateContentResponse(
                candidates=[types.Candidate(grounding_metadata=dodgy_grounding)]
            ),
            False,
        ),
    ],
)
def test_check_grounding_ran(response: types.GenerateContentResponse, expected: bool):
    did_grounding = check_grounding_ran(response)
    assert did_grounding == expected


@mark.parametrize(
    "candidates,text",
    [
        param(None, None, id="no-candidates"),
        param([Mock()], None, id="no-text"),
    ],
)
def test_add_citations_raises_when_missing_output(candidates, text):
    response = Mock(types.GenerateContentResponse)
    response.candidates = candidates
    response.text = text
    response.prompt_feedback = "blocked"
    with raises(GeminiError):
        add_citations(response)
