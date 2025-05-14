"""Tests for the PatientStudyExtractor."""

import json
from pathlib import Path

import pytest

from .example_pipelines.patient_study.model import PatientStudyExtractor
from ns_extract.dataset import Dataset


@pytest.fixture
def sample_data(request) -> Dataset:
    """Load the sample dataset"""
    sample_path = Path("tests/data/sample_inputs")
    return Dataset(sample_path)


@pytest.fixture
def mock_demographics():
    """Create mock demographics data with different group types."""
    sample_path = Path("tests/data/sample_inputs")
    # Get sorted list of study IDs to ensure consistent ordering
    study_ids = sorted([d.name for d in sample_path.iterdir() if d.is_dir()])

    # Create alternating patterns of patient/non-patient studies
    mock_data = {}
    for i, study_id in enumerate(study_ids):
        if i % 2 == 0:
            # Patient study
            mock_data[study_id] = {
                "groups": [
                    {"name": "patient", "age_mean": 45.0 + i},
                    {"name": "control", "age_mean": 44.0 + i},
                ]
            }
        else:
            # Non-patient study
            mock_data[study_id] = {
                "groups": [
                    {"name": "healthy", "age_mean": 40.0 + i},
                    {"name": "control", "age_mean": 39.0 + i},
                ]
            }
    return mock_data


def test_patient_study_extractor(sample_data, mock_demographics, tmp_path):
    """Test patient study extraction."""
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

    # Write results for each study
    for study_id, results in mock_demographics.items():
        study_dir = config_dir / study_id
        study_dir.mkdir(parents=True)

        # Write results and info files
        with open(study_dir / "results.json", "w") as f:
            json.dump(results, f)

        with open(study_dir / "info.json", "w") as f:
            json.dump({"date": "2025-04-19", "valid": True}, f)

    # Initialize extractor
    extractor = PatientStudyExtractor()

    # Set up pipeline info for demographics dependency
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
    output_version_dir = output_dir / "PatientStudyExtractor" / extractor._version
    assert output_version_dir.exists()

    # Get the hash directory
    hash_dir = next(output_version_dir.iterdir())

    # Get list of study IDs
    sample_path = Path("tests/data/sample_inputs")
    study_ids = sorted([d.name for d in sample_path.iterdir() if d.is_dir()])

    # Verify each study's results
    for i, study_id in enumerate(study_ids):
        study_dir = hash_dir / study_id
        results = json.loads((study_dir / "results.json").read_text())
        info = json.loads((study_dir / "info.json").read_text())

        # Every even-numbered study should be a patient study
        assert results["patient_study"] is (i % 2 == 0)
        assert info["valid"] is True

    # Verify pipeline info was written with correct configuration
    pipeline_info = json.loads((hash_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == "1.0.0"
    assert pipeline_info["extractor"] == "PatientStudyExtractor"
    assert pipeline_info["input_pipelines"] == {
        "participant_demographics": {
            "pipeline_dir": str(demographics_dir),
            "version": "1.0.0",
            "config_hash": "abc123",
        }
    }
