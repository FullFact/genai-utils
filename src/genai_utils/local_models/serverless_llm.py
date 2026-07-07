"""
This file contains basic functionality for making requests to a model deployed
as a vast.ai *serverless* endpoint.

Unlike a plain instance (where you'd POST straight to the vLLM/llama.cpp server,
see ``llm.LLM``), a serverless endpoint sits behind vast's autoscaler:
you ask the autoscaler for a ready worker and it routes the request. The
``vastai`` SDK's async ``Serverless`` client handles that for you, so this is an
async client.

Set ``VAST_API_KEY`` in the environment (the SDK reads it; auth is handled by
vast's routing layer, not by the model itself).
"""

import time

from vastai import Serverless

from genai_utils.local_models import NoOutputError
from genai_utils.local_models.throughput import log_tokens_per_second


class ServerlessLLM:
    """
    A model served via a vast.ai serverless endpoint.

    The vast SDK is async, so this is an async client: await
    :meth:`run_message`.
    """

    def __init__(
        self,
        endpoint_name: str,
        model_name: str = "local-model",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Args:
            endpoint_name:
                The name of your vast.ai serverless endpoint.
            model_name:
                The model or LoRA adapter to run (e.g. an adapter name like
                "claimants", or the base model name set when the endpoint was
                deployed).
            max_tokens:
                Default generation length. Also used as the autoscaler's load
                estimate (``cost``) for each request, since generation length
                drives GPU time.
            temperature:
                Default sampling temperature.
        """
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def run_message(self, message_content: str, use_thinking: bool) -> str:
        """
        Sends the message to the serverless endpoint and returns the result.
        Will use thinking if you ask it to.

        The autoscaler waits for a ready worker before routing, so — unlike the
        direct client — there's no "model loading" (503) retry to ride out here.
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

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": str(message_content),
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **extra_model_config,
        }

        start = time.perf_counter()
        async with Serverless() as client:
            endpoint = await client.get_endpoint(name=self.endpoint_name)
            # `cost` is the autoscaler's load estimate for this request;
            # max_tokens is the right proxy since generation length drives GPU
            # time.
            resp = await endpoint.request(
                "/v1/chat/completions",
                payload,
                cost=self.max_tokens,
                stream=False,
            )

        response = resp["response"]
        log_tokens_per_second(response, time.perf_counter() - start)
        response_text = (
            (response.get("choices") or [{}])[0].get("message", {}).get("content")
        )
        if response_text is None:
            raise NoOutputError()
        return response_text
