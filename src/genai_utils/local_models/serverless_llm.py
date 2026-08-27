"""
This file contains basic functionality for making requests to a model deployed
as a vast.ai *serverless* endpoint.

Unlike a plain instance (where you'd POST straight to the vLLM/llama.cpp server,
see ``llm.LLM``), a serverless endpoint sits behind vast's autoscaler:
you ask the autoscaler for a ready worker and it routes the request. The
``vastai`` SDK's async ``Serverless`` client handles that for you, so this is an
async client.

The ``Serverless`` client and the resolved endpoint are built once and reused
across calls (see :meth:`ServerlessLLM._get_endpoint`). Constructing the client
downloads vast's root SSL certificate and opens an aiohttp session, and
resolving the endpoint by name is a control-plane round-trip that lists every
endpoint on the account — doing all of that per request (the previous
behaviour) dominated latency when scoring a document one sentence at a time.

Set ``VAST_API_KEY`` in the environment (the SDK reads it; auth is handled by
vast's routing layer, not by the model itself).
"""

import asyncio
import logging
import time

from vastai import Serverless
from vastai.serverless.client.endpoint import Endpoint_

from genai_utils.local_models import NoOutputError
from genai_utils.local_models.throughput import log_tokens_per_second

_logger = logging.getLogger(__name__)

# Total seconds a request may take, start to finish.
#
# The SDK retries by default, and every time check inside that loop is guarded
# on this being non-None — so passing nothing retries forever, billing GPU time
# on each attempt. Its own checks run between attempts and while polling for a
# worker, never during the worker call itself, which has a separate 600s budget
# we cannot reach through ``request``. So this is enforced from the outside too
# (see :meth:`ServerlessLLM.run_message`), making it a real ceiling.
#
# 300s is generous for a request a warm worker serves. It does not cover waking
# a cold worker, which takes minutes: keep workers warm, or raise it.
DEFAULT_REQUEST_TIMEOUT = 300.0


class ServerlessLLM:
    """
    A model served via a vast.ai serverless endpoint.

    The vast SDK is async, so this is an async client: await
    :meth:`run_message`.

    The underlying ``Serverless`` client and endpoint are created lazily on the
    first request and reused thereafter. Call :meth:`aclose` on shutdown to
    release the client's connection pool.
    """

    def __init__(
        self,
        endpoint_name: str,
        model_name: str = "local-model",
        max_tokens: int = 512,
        temperature: float = 0.7,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
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
            request_timeout:
                Total seconds for the whole call: routing, queueing, retries
                and generation. See :data:`DEFAULT_REQUEST_TIMEOUT`.
        """
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        # Built once on first use and reused across calls (see _get_endpoint).
        self._client: Serverless | None = None
        self._endpoint: Endpoint_ | None = None
        # The vast client's aiohttp session is bound to the event loop it was
        # created on, so we track that loop and rebuild if it changes.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._init_lock = asyncio.Lock()

    async def _get_endpoint(self) -> Endpoint_:
        """Return the resolved endpoint, building and caching the ``Serverless``
        client and endpoint on first use.

        The vast client holds an aiohttp session bound to the event loop it was
        created on, so if the running loop has changed (e.g. a fresh
        ``asyncio.run``) the client is rebuilt against the new loop. The endpoint
        object refreshes its own worker routing over time, so caching it is safe.
        """
        loop = asyncio.get_running_loop()
        if self._client is not None and self._loop is loop:
            assert self._endpoint is not None
            return self._endpoint

        async with self._init_lock:
            # Re-check inside the lock: a concurrent first caller may have built
            # the client while we were waiting for the lock.
            if self._client is not None and self._loop is loop:
                assert self._endpoint is not None
                return self._endpoint

            if self._client is not None:
                # The loop changed under us. We can't await-close a session bound
                # to a now-dead loop, so drop the reference and let it be GC'd.
                _logger.warning(
                    "Event loop changed; rebuilding vast Serverless client "
                    "(old session will be garbage collected)."
                )

            client = Serverless()
            self._endpoint = await client.get_endpoint(name=self.endpoint_name)
            self._client = client
            self._loop = loop
            return self._endpoint

    async def run_message(self, message_content: str, use_thinking: bool) -> str:
        """
        Sends the message to the serverless endpoint and returns the result.
        Will use thinking if you ask it to.

        The autoscaler waits for a ready worker before routing, so — unlike the
        direct client — there's no "model loading" (503) retry to ride out
        here. The SDK does retry routing and connection failures;
        ``request_timeout`` bounds the whole call and raises
        :class:`asyncio.TimeoutError` when spent.
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

        endpoint = await self._get_endpoint()
        start = time.perf_counter()
        # `cost` is the autoscaler's load estimate for this request; max_tokens
        # is the right proxy since generation length drives GPU time.
        #
        # The timeout is passed twice on purpose. The SDK's own budget lets it
        # give up cleanly between retries; wait_for is the outer ceiling, and is
        # what covers a worker call already in flight, since the SDK only checks
        # its budget between attempts.
        resp = await asyncio.wait_for(
            endpoint.request(
                "/v1/chat/completions",
                payload,
                cost=self.max_tokens,
                stream=False,
                timeout=self.request_timeout,
            ),
            timeout=self.request_timeout,
        )

        response = resp["response"]
        log_tokens_per_second(response, time.perf_counter() - start)
        response_text = (
            (response.get("choices") or [{}])[0].get("message", {}).get("content")
        )
        if response_text is None:
            raise NoOutputError()
        return response_text

    async def aclose(self) -> None:
        """Close the underlying vast client's connection pool.

        Safe to call more than once. A fresh client is lazily rebuilt if the
        instance is used again on a live loop.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._endpoint = None
            self._loop = None
