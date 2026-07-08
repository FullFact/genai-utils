import logging

from genai_utils.local_models.throughput import log_tokens_per_second

LOGGER_NAME = "genai_utils.local_models.throughput"


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
