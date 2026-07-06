"""
This file contains basic functionality for making requests to a hosted
OpenAI API compatible LLM.
"""

import logging
import time

import openai
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_logger = logging.getLogger(__name__)


def _is_model_loading(exception: BaseException) -> bool:
    # Servers (vLLM, llama.cpp) return 503 while up but not yet ready to
    # serve — e.g. still loading the model, or a freshly-started replica.
    # Retrying rides out that window; stop_after_attempt bounds the wait.
    return (
        isinstance(exception, openai.InternalServerError)
        and exception.status_code == 503
    )


def log_tokens_per_second(response: object, elapsed: float) -> None:
    """
    Logs generation throughput (tokens/second) at info level.

    Prefers the server's own decode timing when available (llama.cpp reports a
    ``timings`` object with ``predicted_per_second``), otherwise falls back to
    ``completion_tokens`` over the measured wall-clock time, which works for any
    OpenAI-compatible server such as vLLM.
    """
    if not _logger.isEnabledFor(logging.INFO):
        return

    # llama.cpp: authoritative decode rate from the server, no network overhead.
    # The openai client stashes unknown response fields on ``model_extra``.
    extra = getattr(response, "model_extra", None) or {}
    timings = extra.get("timings") if isinstance(extra, dict) else None
    if isinstance(timings, dict) and timings.get("predicted_per_second") is not None:
        _logger.info(
            "Generation throughput: %.1f tok/s (%d tokens in %.2fs, server timings)",
            timings["predicted_per_second"],
            timings.get("predicted_n", 0),
            timings.get("predicted_ms", 0) / 1000,
        )
        return

    # vLLM / generic fallback: completion tokens over wall-clock time.
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens and elapsed > 0:
        _logger.info(
            "Generation throughput: %.1f tok/s (%d completion tokens in %.2fs)",
            completion_tokens / elapsed,
            completion_tokens,
            elapsed,
        )


class NoOutputError(Exception):
    pass


class LLM:
    """
    An LLM using the Open AI API.
    This might be something served via VLLM or Llama.cpp.
    Mostly just wraps the methods for sending requests, etc.
    If the model needs an API key, you will need to provide one.
    """

    def __init__(
        self,
        model_url: str,
        model_name: str = "local-model",
        api_key: str = "not-needed",
    ):
        """
        Args:
            model_url:
                The url where the model is hosted.
            model_name:
                The name of the model.
                This will have been set when you deployed the model.
                The previous hardcoded name was "local-model"
                so that behaviour remains as default.
            api_key:
                The API key for the model, if you need one.
        """
        self.model_url = model_url
        self.model_name = model_name
        self.client = OpenAI(
            base_url=f"{model_url}/v1",
            api_key=api_key,
        )

    @retry(
        retry=retry_if_exception(_is_model_loading),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )
    def run_message(self, message_content: str, use_thinking: bool) -> str:
        """
        Sends the message to the client in the correct formatting.
        Returns the result.
        Will use thinking if you ask it to.
        """
        extra_model_config = (
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_budget": 0,
            }
            if not use_thinking
            else {
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )
        # Ask llama.cpp to report per-request timings. Ignored by servers that
        # don't support it (e.g. vLLM), so it's safe to always send.
        extra_model_config["timings_per_token"] = True

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": str(message_content),
                },
            ],
            extra_body=extra_model_config,
        )
        log_tokens_per_second(response, time.perf_counter() - start)
        response_text = response.choices[0].message.content
        if response_text is None:
            raise NoOutputError()
        return response_text
