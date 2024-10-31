from pathlib import Path
import pytest


@pytest.fixture
def sample_data():
    """Return a sample dataset."""
    return Path(__file__).parents[1] / "tests/data/sample_inputs"
