from pathlib import Path
import pytest
import os

if not os.environ.get("OPENAI_API_KEY", None):
    # This is a test key and should not be used for production
    os.environ["OPENAI_API_KEY"] = "TEST_OPENAI_API_KEY"

@pytest.fixture
def sample_data():
    """Return a sample dataset."""
    return Path(__file__).parents[1] / "tests/data/sample_inputs"

@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "authorization",
            "cookie",
            "user-agent",
            "x-stainless-arch",
            "x-stainless-async",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
        ]
    }
