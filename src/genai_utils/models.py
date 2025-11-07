from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int
    text: str
