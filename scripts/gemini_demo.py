"""
A demo of how to use Gemini in genai utils
"""

from pprint import pp

from genai_utils.gemini import run_prompt
from genai_utils.parsing import parse_model_json_output

# Basic usage
print("BASIC USAGE" + "\n" + "-" * 100)

test_prompt = """
Generate a family tree for the Plantagenet monarchs of England. Output should be in correctly formatted json.
""".strip()

output = run_prompt(test_prompt)
parsed = parse_model_json_output(output)
pp(parsed)


# Multimodal usage
print("\n\nMULTIMODEL" + "\n" + "-" * 100)

video = "gs://raphael-test/tiktok/7149378297489558830.mp4"

output = run_prompt("Summarise this video", video_uri=video)

pp(output)


# Structured output
print("\n\nSTRUCTURED OUTPUT" + "\n" + "-" * 100)
from pydantic import BaseModel, Field

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


# System instructions
print("\n\nSYSTEM INSTRUCTIONS\n", "-" * 100)

prompt = "Write a one paragraph review of Jaws."
instruction = (
    "You are a film critic."
    "You hate Hollywood films."
    "And you constantly mention your pet Badger."
)

output = run_prompt(prompt, system_instruction=instruction)
pp(output)


# Grounding
prompt = (
    "Who won the Battle of Crécy, and what was it's historical significance?"
    "Answer in one paragraph."
)

answer_inline_citations = run_prompt(prompt, use_grounding=True, inline_citations=True)
print("GROUNDING W/ CITATIONS", "\n", "-" * 100, "\n", answer_inline_citations, "\n\n")
