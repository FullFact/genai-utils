import logging

from genai_utils.llm.throughput import log_token_usage, log_tokens_per_second

LOGGER_NAME = "genai_utils.llm.throughput"


def test_log_tokens_per_second_silent_when_info_disabled(caplog) -> None:
    """Should be a no-op (and never raise) when info logging is off."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    log_tokens_per_second({}, elapsed=1.0)
    assert caplog.records == []


def test_log_tokens_per_second_prefers_server_timings(caplog) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    response = {
        "timings": {
            "predicted_per_second": 42.0,
            "predicted_n": 10,
            "predicted_ms": 250,
        },
        "usage": {"completion_tokens": 999},
    }
    log_tokens_per_second(response, elapsed=5.0)

    assert "throughput" in caplog.text
    # the server timing rate wins over the wall-clock fallback
    assert "42.0 tok/s" in caplog.text


def test_log_tokens_per_second_falls_back_to_usage(caplog) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    response = {"usage": {"completion_tokens": 20}}
    log_tokens_per_second(response, elapsed=2.0)

    # 20 tokens over 2 seconds => 10 tok/s
    assert "10.0 tok/s" in caplog.text


def test_log_token_usage_logs_prompt_and_completion(caplog) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    response = {"usage": {"prompt_tokens": 1234, "completion_tokens": 56}}
    log_token_usage(response, model_name="topics")

    assert (
        "LLM token usage: model=topics prompt_tokens=1234 completion_tokens=56"
        in caplog.text
    )
    record = caplog.records[0]
    assert record.model == "topics"
    assert record.prompt_tokens == 1234
    assert record.completion_tokens == 56


def test_log_token_usage_silent_without_usage(caplog) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    log_token_usage({}, model_name="topics")
    assert caplog.records == []


def test_log_token_usage_silent_when_info_disabled(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    log_token_usage({"usage": {"prompt_tokens": 1}}, model_name="topics")
    assert caplog.records == []
