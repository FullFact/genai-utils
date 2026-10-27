# Generative AI Utilities

This repo contains utility functions for using Generative AI models.

Broadly it contains functionality for:
1. Running prompts through LLMs
2. Parsing the json output from LLMs

It supports Gemini, models served behind an OpenAI-compatible API (e.g. vLLM or llama.cpp), and vast.ai serverless endpoints.

## Setup

If you don't want to manually specify the config of Gemini, you should set the following environment variables:
* `GEMINI_PROJECT`: the GCP project you want to use Gemini in, e.g. "my-project"
* `GEMINI_LOCATION`: the GCP location you want to run Gemini on, e.g. "global"
* `GEMINI_MODEL`: the Gemini model you wish to use, e.g. "gemini-2.5-flash-lite"

## Basic usage
Here's a simple demo of how you can use it:
```python
from pprint import pp
from genai_utils.llm.gemini import GeminiChat
from genai_utils.parsing import parse_model_json_output

test_prompt = """
Generate a family tree for the Plantagenet monarchs of England. Output should be in correctly formatted json.
""".strip()

model = GeminiChat()
output = model.run_message_sync(test_prompt, use_thinking=False)
parsed = parse_model_json_output(output)
pp(parsed)
```

## Multimodal

It can also be used to process videos.
```python
video = "gs://raphael-test/tiktok/7149378297489558830.mp4"

output = model.run_message_sync("Summarise this video", use_thinking=False, video_uri=video)

pp(output)
```

You can of course swap that uri for another video uri.

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


model = GeminiChat(output_schema=list[Monarch])
output = model.run_message_sync(test_prompt, use_thinking=False)
parsed = parse_model_json_output(output)
pp(parsed)
```

As you see there, you just need to make a pydantic `BaseModel` and give that to `GeminiChat`.
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

model = GeminiChat(system_instruction=instruction)
output = model.run_message_sync(prompt, use_thinking=False)
pp(output)
```

## Grounding

You can get Gemini to do a Google search, which it will then use to make its response.
This means you can access more up to date information, get a more factual response, and get citations.
It will be a significantly more expensive query though.

You can control this using the `use_grounding` and `inline_citations` arguments of `GeminiChat`.
Inline citations only work if `use_grounding=True`.
If `inline_citations=False` Gemini will still do the search and use the evidence, but it won't include the links in the text.
Currently grounding does not work with structured output, so you must choose between grounding and structured output.

Here's an example:
```python
prompt = (
    "Who won the Battle of Crécy, and what was it's historical significance?"
    "Answer in one paragraph."
)

model = GeminiChat(use_grounding=True, inline_citations=True)
result = model.run_message_sync(prompt, use_thinking=False)
pp(result)
```

## Thinking

You can enable or disable thinking in the model by toggling the `use_thinking` parameter.
Only enable this if the task is complex enough to require it, because it makes things slow and expensive.

## Async and switching models

`run_message` is the async version of `run_message_sync`.
`GeminiChat`, `OpenaiChat` and `ServerlessChat` all implement the `ChatModel` protocol, so code that takes a `ChatModel` works with any of them:

```python
from genai_utils.llm.models import ChatModel
from genai_utils.llm.openai import OpenaiChat


async def summarise(model: ChatModel, text: str) -> str:
    return await model.run_message(f"Summarise this: {text}", use_thinking=False)


model = OpenaiChat(model_url="http://localhost:8080")
summary = await summarise(model, "...")
await model.aclose()
```