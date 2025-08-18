import os
from pathlib import Path

import pytest
import requests


@pytest.fixture
def sample_data():
    """Return a sample dataset."""
    return Path(__file__).parents[1] / "tests/data/sample_inputs"


@pytest.fixture(scope="session")
def setup_tiktoken_cache():
    """
    Ensure tiktoken finds encoding files in tests.
    Downloads cl100k_base.tiktoken to data/tiktoken if not present.
    """
    local_vocab_dir = Path(os.path.dirname(__file__)) / "data" / "tiktoken"
    os.environ["TIKTOKEN_CACHE_DIR"] = str(local_vocab_dir)


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
    }
