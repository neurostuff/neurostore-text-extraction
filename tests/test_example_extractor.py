"""Tests for the ExampleExtractor."""

import json
from pathlib import Path

import pytest

from .example_pipelines.example_extractor.model import ExampleExtractor
from ns_extract.dataset import Dataset


@pytest.fixture
def sample_data(request) -> Dataset:
    """Load the sample dataset"""
    sample_path = Path("tests/data/sample_inputs")
    return Dataset(sample_path)


@pytest.fixture
def mock_demographics():
    """Create mock demographics data."""
    sample_path = Path("tests/data/sample_inputs")
    study_ids = sorted([d.name for d in sample_path.iterdir() if d.is_dir()])

    mock_data = {}
    for i, study_id in enumerate(study_ids):
        mock_data[study_id] = {
            "groups": [
                {"name": f"group_{i}_a", "age_mean": 45.0 + i},
                {"name": f"group_{i}_b", "age_mean": 44.0 + i},
            ]
        }
    return mock_data


def test_example_extractor(sample_data, mock_demographics, tmp_path):
    """Test example extraction."""
    # Create demographics pipeline outputs
    demographics_dir = tmp_path / "participant_demographics"
    version_dir = demographics_dir / "1.0.0"
    config_dir = version_dir / "abc123"
    config_dir.mkdir(parents=True)

    # Write pipeline info
    with open(config_dir / "pipeline_info.json", "w") as f:
        json.dump(
            {
                "date": "2025-04-19",
                "version": "1.0.0",
                "type": "participant_demographics",
            },
            f,
        )

    # Write demographics results for each study
    for study_id, results in mock_demographics.items():
        study_dir = config_dir / study_id
        study_dir.mkdir(parents=True)

        # Write results and info files
        with open(study_dir / "results.json", "w") as f:
            json.dump(results, f)

        with open(study_dir / "info.json", "w") as f:
            json.dump({"date": "2025-04-19", "valid": True}, f)

    # Initialize extractor
    extractor = ExampleExtractor()

    # Set up pipeline info for dependencies
    input_pipeline_info = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": Path(demographics_dir),
        }
    }

    # Run extraction
    output_dir = tmp_path / "output"
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Check results for each study
    output_version_dir = output_dir / "ExampleExtractor" / extractor._version
    assert output_version_dir.exists()

    # Get the hash directory
    hash_dir = next(output_version_dir.iterdir())
    assert hash_dir.is_dir()

    # Get list of study IDs
    sample_path = Path("tests/data/sample_inputs")
    study_ids = sorted([d.name for d in sample_path.iterdir() if d.is_dir()])

    # Verify each study's results exist and are properly formatted
    for study_id in study_ids:
        study_dir = hash_dir / study_id
        assert study_dir.exists()

        # Check for required files
        assert (study_dir / "results.json").exists()
        assert (study_dir / "info.json").exists()

        # Verify content can be loaded
        results = json.loads((study_dir / "results.json").read_text())
        info = json.loads((study_dir / "info.json").read_text())

        # Verify info file structure
        assert "date" in info
        assert "valid" in info
        assert info["valid"] is True

    # Verify pipeline info was written with correct configuration
    pipeline_info = json.loads((hash_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == extractor._version
    assert pipeline_info["extractor"] == "ExampleExtractor"
    assert pipeline_info["input_pipelines"] == {
        "participant_demographics": {
            "pipeline_dir": str(demographics_dir),
            "version": "1.0.0",
            "config_hash": "abc123"
        }
    }
