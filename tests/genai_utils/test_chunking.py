from pytest import mark, param, raises

from genai_utils.chunking import (
    calculate_n_chunks,
    make_chunks,
)


@mark.parametrize(
    "n_sentences,chunk_size,overlap_size,expected_chunk_count",
    [
        param(50, 100, 10, 1, id="short transcript with overlap"),
        param(350, 100, 10, 4, id="long transcript with overlap"),
        param(50, 100, 0, 1, id="short transcript without overlap"),
        param(350, 100, 0, 4, id="long transcript without overlap"),
        param(350, 100, 50, 6, id="long transcript with big overlap"),
        param(10, 100, 15, 1, id="text shorter than overlap"),
        param(50, 50, 0, 1, id="sentences same as chunk size"),
        param(20, 50, 0, 1, id="sentences shorter than chunk size"),
        param(20, 19, 0, 2, id="sentences one longer than chunk size"),
        param(0, 100, 0, 0, id="no sentences"),
    ],
)
def test_calculate_n_chunks(
    n_sentences, chunk_size, overlap_size, expected_chunk_count
) -> None:
    assert (
        calculate_n_chunks(n_sentences, chunk_size, overlap_size)
        == expected_chunk_count
    )


@mark.parametrize(
    "n_sentences,chunk_size,overlap_size",
    [
        param(50, 100, 10, id="short transcript with overlap"),
        param(350, 100, 10, id="long transcript with overlap"),
        param(350, 100, 50, id="long transcript with big overlap"),
        param(20, 19, 0, id="sentences one longer than chunk size"),
        param(0, 100, 0, id="no sentences"),
    ],
)
def test_make_chunks_count_matches_calculation(
    n_sentences, chunk_size, overlap_size
) -> None:
    """``make_chunks`` should yield exactly as many chunks as predicted."""
    sentences = [f"sentence {i}" for i in range(n_sentences)]
    chunks = list(make_chunks(sentences, chunk_size, overlap_size))
    assert len(chunks) == calculate_n_chunks(n_sentences, chunk_size, overlap_size)


def test_make_chunks_overlap_content() -> None:
    """Consecutive chunks should share ``overlap_size`` sentences."""
    sentences = [f"s{i}" for i in range(10)]
    chunks = list(make_chunks(sentences, chunk_size=4, overlap_size=2))

    # first chunk is the start of the input
    assert chunks[0] == ["s0", "s1", "s2", "s3"]
    # the tail of one chunk is the head of the next
    assert chunks[0][-2:] == chunks[1][:2]
    assert chunks[1] == ["s2", "s3", "s4", "s5"]


def test_make_chunks_no_overlap_partitions_input() -> None:
    """With no overlap the chunks should join back into the original list."""
    sentences = [f"s{i}" for i in range(10)]
    chunks = list(make_chunks(sentences, chunk_size=3, overlap_size=0))
    flattened = [sentence for chunk in chunks for sentence in chunk]
    assert flattened == sentences


def test_make_chunks_no_redundant_trailing_chunk() -> None:
    """A final chunk fully contained in the previous one should not be emitted."""
    sentences = [f"s{i}" for i in range(10)]
    # chunk_size=4, overlap=2, step=2 would step to index 8 and yield a final
    # ["s8", "s9"] that is already wholly inside the previous ["s6"..."s9"] chunk.
    chunks = list(make_chunks(sentences, chunk_size=4, overlap_size=2))
    assert chunks[-1] == ["s6", "s7", "s8", "s9"]
    # every chunk carries new content the previous one didn't have
    assert all(chunks[i] != chunks[i - 1] for i in range(1, len(chunks)))


def test_make_chunks_empty_input() -> None:
    assert list(make_chunks([], chunk_size=5, overlap_size=2)) == []


@mark.parametrize(
    "chunk_size,overlap_size",
    [
        param(10, 15, id="overlap bigger than chunk"),
        param(10, 10, id="overlap same as chunk"),
        param(100, -1, id="negative overlap"),
    ],
)
def test_make_chunks_invalid_overlap_raises(chunk_size, overlap_size) -> None:
    with raises(ValueError):
        # wrap in list to force the generator to run
        list(make_chunks(["x"] * 50, chunk_size, overlap_size))
