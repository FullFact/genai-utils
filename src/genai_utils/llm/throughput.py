import logging
from typing import Any

_logger = logging.getLogger(__name__)

TOKEN_USAGE_EVENT_METRIC = "llm_token_usage"


def log_tokens_per_second(response: dict[str, Any], elapsed: float) -> None:
    """
    Logs generation throughput (tokens/second) at info level.

    Takes a plain OpenAI-shaped ``dict``. The serverless endpoint returns one
    directly; the direct :class:`OpenaiChat` client passes ``response.model_dump()``.
    Prefers the server's own decode timing when available (llama.cpp reports a
    ``timings`` object with ``predicted_per_second``), otherwise falls back to
    ``completion_tokens`` over the measured wall-clock time, which works for any
    OpenAI-compatible server such as vLLM.
    """
    if not _logger.isEnabledFor(logging.INFO):
        return

    # llama.cpp: authoritative decode rate from the server, no network overhead.
    timings = response.get("timings")
    if isinstance(timings, dict) and timings.get("predicted_per_second") is not None:
        _logger.info(
            "Generation throughput: %.1f tok/s (%d tokens in %.2fs, server timings)",
            timings["predicted_per_second"],
            timings.get("predicted_n", 0),
            timings.get("predicted_ms", 0) / 1000,
        )
        return

    # vLLM / generic fallback: completion tokens over wall-clock time.
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens and elapsed > 0:
        _logger.info(
            "Generation throughput: %.1f tok/s (%d completion tokens in %.2fs)",
            completion_tokens / elapsed,
            completion_tokens,
            elapsed,
        )


def log_token_usage(response: dict[str, Any], model_name: str) -> None:
    """
    Logs the token counts of one call at info level.

    Reads the ``usage`` block of an OpenAI-shaped response, which vLLM and
    llama.cpp both return. The message is fixed so a log-based metric can match
    it; the counts go in ``extra`` as ``model``, ``prompt_tokens`` and
    ``completion_tokens``, which structured log handlers emit as fields.
    ``reasoning_tokens`` is added when the server reports it in
    ``usage.completion_tokens_details``, as OpenAI and some vLLM reasoning
    parsers do. Logs nothing if the server sent no ``usage``.
    """
    if not _logger.isEnabledFor(logging.INFO):
        return

    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return

    fields: dict[str, Any] = {
        "event_metric": TOKEN_USAGE_EVENT_METRIC,
        "model": model_name,
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
    }
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is not None:
        fields["reasoning_tokens"] = reasoning_tokens

    _logger.info("llm token usage", extra=fields)
