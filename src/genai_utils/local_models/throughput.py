import logging
from typing import Any

_logger = logging.getLogger(__name__)


def log_tokens_per_second(response: dict[str, Any], elapsed: float) -> None:
    """
    Logs generation throughput (tokens/second) at info level.

    Takes a plain OpenAI-shaped ``dict``. The serverless endpoint returns one
    directly; the direct :class:`LLM` client passes ``response.model_dump()``.
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
