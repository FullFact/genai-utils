from pytest import mark, param, raises

from genai_utils.parsing import parse_model_json_output


@mark.parametrize(
    "model_output,expected,should_succeed",
    [
        param(
            """
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"}
            ]
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            True,
            id="perfect case",
        ),
        param(
            """
            json```
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"}
            ]```
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            True,
            id="prefixed with json, surrounded by backticks",
        ),
        param(
            """
            json```
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"}
            ]
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            True,
            id="missing final backticks",
        ),
        param(
            """
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"}
            ]```
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            True,
            id="only backticks at the end",
        ),
        param(
            """
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"}
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            True,
            id="missing closing brace",
        ),
        param(
            """
                {"claim": "claim 1"},
                {"claim": "claim 2"}
            ]
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            True,
            id="missing opening brace",
        ),
        param(
            """
            Claim: Claim 1
            Claim: Claim 2
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            False,
            id="not json",
        ),
        param(
            """
            [
                {'claim': 'claim 1'},
                {'claim': 'claim 2'}
            ]
            """,
            [
                {"claim": "claim 1"},
                {"claim": "claim 2"},
            ],
            True,
            id="single quotes",
        ),
        param(
            """
            []
            """,
            [],
            True,
            id="Empty list",
        ),
    ],
)
def test_parse_model_json_output(
    model_output: str, expected: list[dict[str, str]], should_succeed: bool
) -> None:
    if should_succeed:
        assert parse_model_json_output(model_output) == expected
    else:
        with raises(ValueError):
            assert parse_model_json_output(model_output)
