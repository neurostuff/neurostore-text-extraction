from pathlib import Path
import pytest


@pytest.fixture
def sample_data():
    """Return a sample dataset."""
    return Path(__file__).parents[1] / "tests/data/sample_inputs"


@pytest.fixture(scope="module", autouse=True)
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
        ],
        "ignore_hosts": ["blob.core.windows.net"],
    }
