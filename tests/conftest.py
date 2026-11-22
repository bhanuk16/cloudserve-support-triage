"""Test configuration. Tests run fully offline with the deterministic mock model, in an
isolated temporary storage directory, so they never need an API key and never touch
your real decision log or cache."""
import os
import shutil
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="cst-tests-"))
os.environ.update({
    "LLM_PROVIDER": "mock",
    "LLM_API_KEY": "",
    "DECISION_DB_PATH": str(_TMP / "decisions.db"),
    "LLM_CACHE_PATH": str(_TMP / "cache.db"),
    "KILL_SWITCH_FILE": str(_TMP / "KILL_SWITCH"),
    "CHROMA_PATH": str(_TMP / "chroma"),
    "AUTO_RESPOND_ENABLED": "true",
})

import pytest  # noqa: E402

from src.config import settings  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ready():
    """Install the embedding model and train the classifier if a clean checkout lacks them."""
    from evaluation.harness import preflight
    preflight()
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def pipe():
    from src.pipeline import Pipeline
    return Pipeline()


@pytest.fixture
def ticket():
    from src.ingest import normalise

    def make(body, channel="email", subject="", **kw):
        return normalise({"ticket_id": kw.pop("ticket_id", "T-1"), "channel": channel, "subject": subject,
                          "body": body, "customer_name": kw.pop("customer_name", "Priya Sharma"), **kw})
    return make


@pytest.fixture
def tmp_storage():
    return _TMP
