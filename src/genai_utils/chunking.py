"""
We frequently end up chunking input sentences when we do things with Gen AI.
This has become replicated across multiple projects,
so we made some laughably simple helper functions here.
"""

import math
from typing import Iterator


def calculate_n_chunks(n_sentences: int, chunk_size: int, overlap_size: int) -> int:
    """
    Calculates the number of chunks that ``make_chunks`` will produce.

    Parameters
    ----------
    n_sentences: int
        Total number of sentences.
    chunk_size: int
        Number of sentences per chunk.
    overlap_size: int
        Number of overlapping sentences between consecutive chunks.

    Returns
    -------
    int
        The number of chunks.
    """
    if n_sentences == 0:
        return 0
    # The first chunk that reaches the end of the input is the last one we
    # emit; any further chunk would be a subset of it (see ``make_chunks``).
    if n_sentences <= chunk_size:
        return 1
    step = chunk_size - overlap_size
    return math.ceil((n_sentences - chunk_size) / step) + 1


def make_chunks(
    sentences: list[str], chunk_size: int, overlap_size: int
) -> Iterator[list[str]]:
    """
    Takes a list of sentences, and yields smaller chunks of sentences,
    with overlaps between chunks.

    Parameters
    ----------
    text: list[str]
        List of sentences to be chunked.
    chunk_size: int
        Number of sentences to be in each chunk.
    overlap_size: int
        Size of overlap of consecutive chunks.

    Returns
    -------
    Iterator[list[str]]
        Yields one chunk of sentences at a time.

    Raises
    ------
    ValueError:
        if the overlap is negative or the overlap is larger than the chunk size.
    """
    if not (0 <= overlap_size < chunk_size):
        raise ValueError(
            f"overlap_size ({overlap_size}) must be >= 0 and "
            f"< chunk_size ({chunk_size})"
        )
    step = chunk_size - overlap_size
    for i in range(0, len(sentences), step):
        yield sentences[i : i + chunk_size]
        # Once a chunk reaches the end, stop: any later chunk would start
        # inside this one's overlap and be fully contained in it.
        if i + chunk_size >= len(sentences):
            break
