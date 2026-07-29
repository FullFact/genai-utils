import pytest

from genai_utils import gemini


@pytest.fixture(autouse=True)
def reset_gemini_clients():
    """
    Empties the shared Gemini client caches around every test.

    `gemini` keeps clients keyed by project/location (and, for async, by event
    loop) so that a process builds one rather than one per prompt. Tests patch
    `genai.Client`, so without this a client built from one test's mock would be
    handed to the next test.
    """
    gemini._sync_clients.clear()
    gemini._async_clients.clear()
    yield
    gemini._sync_clients.clear()
    gemini._async_clients.clear()
