import json_repair
from json_repair.json_parser import JSONReturnType

ParsedType = JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]]


def parse_model_json_output(model_output: str) -> ParsedType:
    parsed = json_repair.loads(model_output)
    if not parsed and parsed != []:
        raise ValueError("Could not parse the string: ", model_output)
    return parsed


if __name__ == "__main__":
    import json
    from pprint import pp

    from genai_utils.gemini import run_prompt

    test_prompt = """
    Generate a family tree for the Plantagenet monarchs of England. Output should be in correctly formatted json.
    """.strip()

    output = run_prompt(test_prompt)
    print(f"Model output: {json.dumps(output)[:100]}")

    parsed = parse_model_json_output(output)
    pp(parsed)
