import logging

from rapidfuzz import fuzz

from genai_utils.models import Span


def find_quote_in_sentence(
    sentence: str, quote: str, threshold: float = 80
) -> Span | None:
    """
    Returns a span for the quote within the sentence if there's a match.

    Parameters
    ----------
    sentence: str
        The sentence to search inside
    quote: str
        The string to find inside the sentence

    Returns
    -------
    Span:
        The span containing the quote.
    """
    similarity = fuzz.partial_ratio_alignment(quote.lower(), sentence.lower())
    if similarity is not None and similarity.score >= threshold:
        span_start = similarity.dest_start
        span_end = similarity.dest_end
        return Span(
            start=span_start,
            end=span_end,
            text=sentence[span_start:span_end],
        )
    return None


def get_best_matching_sentence_for_quote(
    original_quote: str, sentences: list[str]
) -> tuple[int, Span, float] | None:
    """
    Finds the best matching sentence for the given quote out of the provided sentences.

    Parameters
    ----------
    original_quote: str
        The text you want to find a match for.
        In most use cases, this would be the quote made by the Gen AI model.
    sentences: list[str]
        A list of sentences to search.

    Returns
    -------
    tuple[int, Span, float] | None
        A tuple representing a single match, containing the following elements:
        - Index of the matching sentence
        - Span of the quote within said sentence
        - Similarity of the sentence to the quote
    """
    # Get the spans for any sentences matching the claim
    sentences_matching_quote = [
        (sentence_idx, span, fuzz.ratio(original_quote, sentence))
        for sentence_idx, sentence in enumerate(sentences)
        if (span := find_quote_in_sentence(sentence, original_quote)) is not None
    ]

    # return None if there's no matching sentences
    if not len(sentences_matching_quote):
        logging.debug(
            f'No matches found for current quote: "{original_quote[:100]}..."'
        )
        return None
    # get the sentence with the highest overlap (usually the longest)
    return max(sentences_matching_quote, key=lambda x: x[2])


def link_quotes_and_sentences(
    quotes: list[str], sentences: list[str]
) -> list[tuple[int, int, Span]]:
    """
    Links pairs of matching quotes and sentences.
    Each quote given will only match to the single best matching sentence from the list.
    Each sentence may match to multiple quotes.

    Parameters
    ----------
    quotes: list[str]
        A list of quotes to find matches for.
    sentences: list[str]
        A list of sentences to search against each quote.

    Returns
    -------
    list[tuple[int, int, Span]]:
        A list of quote, sentence pairs.
        Each row is a tuple containing the following elements:
        - Index of the quote (from the provided list)
        - Index of the sentence (from the provided list)
        - Span of the match (within the sentence)
    """
    return [
        (quote_idx, best_sentence[0], best_sentence[1])
        for quote_idx, quote in enumerate(quotes)
        if (best_sentence := get_best_matching_sentence_for_quote(quote, sentences))
        is not None
    ]
