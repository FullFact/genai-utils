# Generative AI Utilities

This repo contains utility functions for using Generative AI models.

Broadly it contains functionality for:
1. Running prompts through LLMs
2. Parsing the json output from LLMs

Currently it only supports Gemini, but over time it can support other models.

## Setup

This library uses the [Gemini Developer API](https://ai.google.dev/gemini-api/docs).

Authentication is handled by the SDK via an API key:
* `GEMINI_API_KEY` (or `GOOGLE_API_KEY`): your Gemini API key

If you don't want to manually specify the config of Gemini, you should also set:
* `GEMINI_MODEL`: the Gemini model you wish to use, e.g. "gemini-2.5-flash-lite"

## Basic usage
Here's a simple demo of how you can use it:
```python
from pprint import pp
from genai_utils.gemini import run_prompt
from genai_utils.parsing import parse_model_json_output

test_prompt = """
Generate a family tree for the Plantagenet monarchs of England. Output should be in correctly formatted json.
""".strip()

output = run_prompt(test_prompt)
parsed = parse_model_json_output(output)
pp(parsed)
```

## Multimodal

It can also be used to process videos.
`video_uri` accepts any of:
* A YouTube URL (passed straight to Gemini)
* A local file path (uploaded via the Files API)
* A `gs://` GCS URI — downloaded and then uploaded via the Files API
  (requires the `gcs` extra: `pip install genai-utils[gcs]`)

```python
output = run_prompt("Summarise this video", video_uri="path/to/video.mp4")

pp(output)
```

## Structured Output

You can also get structured output from Gemini by specifying an output schema.

```python
test_prompt = """
List all of the Stewart monarchs of England.
""".strip()


class Monarch(BaseModel):
    birth_name: str = Field(description="Birth name of the monarch")
    monarch_name: str = Field(description="Name as monarch")
    reign_start: str = Field(description="First year of reign")
    reign_end: str = Field(description="Final year of reign")
    spouse: list[str] = Field(description="A list of their spouses")
    children: list[str] = Field(description="A list of their children")


output = run_prompt(test_prompt, output_schema=list[Monarch])
parsed = parse_model_json_output(output)
pp(parsed)
```

As you see there, you just need to make a pydantic `BaseModel` and give that to `run_prompt`.
The types and descriptions will also be passed along with the keys for your json.
In this case we've passed along `list[Monarch]`, because we want a list of `Monarch` dictionaries to be returned.

## System Instructions

You can define a system instruction which basically gets added to the front of the prompt.
This is good for defining the desired behaviour of the model.

```python
prompt = "Write a one paragraph review of Jaws."
instruction = (
    "You are a film critic."
    "You hate Hollywood films."
    "And you constantly mention your pet Badger."
)

output = run_prompt(prompt, system_instruction=instruction)
pp(output)
```

## Grounding

You can get Gemini to do a Google search, which it will then use to make its response.
This means you can access more up to date information, get a more factual response, and get citations.
It will be a significantly more expensive query though.

You can control this using the `use_grounding` and `inline_citations` arguments of `run_prompt`.
Inline citations only work if `use_grounding=True`.
If `inline_citations=False` Gemini will still do the search and use the evidence, but it won't include the links in the text.
Currently grounding does not work with structured output, so you must choose between grounding and structured output.

Here's an example:
```python
prompt = (
    "Who won the Battle of Crécy, and what was it's historical significance?"
    "Answer in one paragraph."
)

result = run_prompt(prompt, use_grounding=True, inline_citations=True)
pp(result)
```

## Thinking

You can enable or disable thinking in the model by toggling the `do_thinking` parameter.
Only enable this if the task is complex enough to require it, because it makes things slow and expensive.