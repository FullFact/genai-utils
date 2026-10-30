"""
Retries for the chat models.

A client opts in by implementing :class:`Retryable` (how many attempts to make,
and which errors are worth another attempt) and decorating its request method
with :func:`retry_on_failure` (or :func:`retry_on_failure_sync`).
Each client knows which of its own errors are transient, so it decides what
to retry; the decorators decide how.
"""

import functools
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from genai_utils.llm.models import describe_model

_logger = logging.getLogger(__name__)

# Lets log-based metrics match the retry warning without relying on its text.
RETRY_EVENT_METRIC = "model_retry"

# Short, as these are for brief blips (a dropped connection, a rate limit),
# not for waiting out a model that is still loading.
# Read at call time so tests can swap it out.
_WAIT: wait_base = wait_exponential(multiplier=1, min=1, max=10)


class Retryable(Protocol):
    """
    A client whose requests can be retried by :func:`retry_on_failure`.
    """

    n_attempts: int
    """How many times to attempt a request before failing. 1 means no retries."""

    def is_retryable(self, exc: BaseException) -> bool:
        """
        Whether ``exc`` is transient and so worth another attempt.
        Errors that would fail the same way again should return False.
        """
        ...


R = TypeVar("R", bound=Retryable)
P = ParamSpec("P")
T = TypeVar("T")


def check_n_attempts(n_attempts: int) -> int:
    """
    Returns ``n_attempts`` if it is valid, for use in a client's ``__init__``.

    Raises
    ------
    ValueError:
        if ``n_attempts`` is less than 1.
    """
    if n_attempts < 1:
        raise ValueError(f"n_attempts must be at least 1, got {n_attempts}")
    return n_attempts


def _log_retry(client: Retryable, state: RetryCallState) -> None:
    """
    Warns before each retry, with ``event_metric``, ``model``, ``attempt`` and
    ``error_type`` as log record attributes, like the fallback warning.
    """
    exc = state.outcome.exception() if state.outcome else None
    sleep = state.next_action.sleep if state.next_action else 0.0
    _logger.warning(
        f"Attempt {state.attempt_number} of {client.n_attempts} failed with "
        f"{exc!r}; retrying in {sleep:.1f}s.",
        extra={
            "event_metric": RETRY_EVENT_METRIC,
            "model": describe_model(client),
            "attempt": state.attempt_number,
            "error_type": type(exc).__name__,
        },
    )


def _retry_settings(client: Retryable) -> dict[str, Any]:
    return {
        "stop": stop_after_attempt(client.n_attempts),
        "retry": retry_if_exception(client.is_retryable),
        "wait": _WAIT,
        "before_sleep": functools.partial(_log_retry, client),
        "reraise": True,
    }


def retry_on_failure(
    method: Callable[Concatenate[R, P], Coroutine[Any, Any, T]],
) -> Callable[Concatenate[R, P], Coroutine[Any, Any, T]]:
    """
    Retries an async method up to ``self.n_attempts`` times,
    on errors that ``self.is_retryable`` accepts.
    The last error is raised once the attempts run out.
    """

    @functools.wraps(method)
    async def wrapper(self: R, *args: P.args, **kwargs: P.kwargs) -> T:
        async for attempt in AsyncRetrying(**_retry_settings(self)):
            with attempt:
                return await method(self, *args, **kwargs)
        raise AssertionError("unreachable: tenacity reraises the last error")

    return wrapper


def retry_on_failure_sync(
    method: Callable[Concatenate[R, P], T],
) -> Callable[Concatenate[R, P], T]:
    """
    A synchronous version of :func:`retry_on_failure`.
    """

    @functools.wraps(method)
    def wrapper(self: R, *args: P.args, **kwargs: P.kwargs) -> T:
        for attempt in Retrying(**_retry_settings(self)):
            with attempt:
                return method(self, *args, **kwargs)
        raise AssertionError("unreachable: tenacity reraises the last error")

    return wrapper
