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
        pipeline_info = {
            "date": "2025-04-19",
            "version": "1.0.0",
            "config_hash": "abc123",
            "extractor": "DemographicsExtractor",
            "extractor_kwargs": {},
            "transform_kwargs": {},
            "input_pipelines": {},
            "schema": {},
        }
        json.dump(pipeline_info, f)

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


def test_no_changes_on_identical_data(sample_data, mock_demographics, tmp_path):
    """Test that transform_dataset produces identical results when called twice."""
    # Setup initial demographics data
    demographics_dir = tmp_path / "participant_demographics"
    version_dir = demographics_dir / "1.0.0"
    config_dir = version_dir / "abc123"
    config_dir.mkdir(parents=True)

    with open(config_dir / "pipeline_info.json", "w") as f:
        pipeline_info = {
            "date": "2025-04-19",
            "version": "1.0.0",
            "config_hash": "abc123",
            "extractor": "DemographicsExtractor",
            "extractor_kwargs": {},
            "transform_kwargs": {},
            "input_pipelines": {},
            "schema": {},
        }
        json.dump(pipeline_info, f)

    for study_id, results in mock_demographics.items():
        study_dir = config_dir / study_id
        study_dir.mkdir(parents=True)
        with open(study_dir / "results.json", "w") as f:
            json.dump(results, f)
        with open(study_dir / "info.json", "w") as f:
            json.dump({"date": "2025-04-19", "valid": True}, f)

    # Initialize extractor and pipeline info
    extractor = PatientStudyExtractor()
    input_pipeline_info = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": Path(demographics_dir),
        }
    }

    # First transformation
    output_dir = tmp_path / "output"
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Store first run results
    first_run_results = {}
    output_version_dir = output_dir / "PatientStudyExtractor" / extractor._version
    hash_dir = next(output_version_dir.iterdir())
    for study_dir in hash_dir.iterdir():
        if study_dir.is_dir():
            first_run_results[study_dir.name] = json.loads(
                (study_dir / "results.json").read_text()
            )

    # Second transformation
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Compare results
    hash_dir = next(output_version_dir.iterdir())
    for study_dir in hash_dir.iterdir():
        if study_dir.is_dir():
            second_run_results = json.loads((study_dir / "results.json").read_text())
            assert second_run_results == first_run_results[study_dir.name]


def test_new_study_addition(sample_data, mock_demographics, tmp_path):
    """Test that transform_dataset correctly reflects new study addition in second call."""
    # Setup initial demographics data
    demographics_dir = tmp_path / "participant_demographics"
    version_dir = demographics_dir / "1.0.0"
    config_dir = version_dir / "abc123"
    config_dir.mkdir(parents=True)

    with open(config_dir / "pipeline_info.json", "w") as f:
        pipeline_info = {
            "date": "2025-04-19",
            "version": "1.0.0",
            "config_hash": "abc123",
            "extractor": "DemographicsExtractor",
            "extractor_kwargs": {},
            "transform_kwargs": {},
            "input_pipelines": {},
            "schema": {},
        }
        json.dump(pipeline_info, f)

    # Get list of study IDs
    study_ids = list(mock_demographics.keys())
    removed_study_id = study_ids[0]  # Take first study to remove and add back
    remaining_study_ids = study_ids[1:]

    # Create reduced dataset without the first study
    reduced_dataset = sample_data.slice(remaining_study_ids)

    # Write demographics data for remaining studies
    for study_id in remaining_study_ids:
        study_dir = config_dir / study_id
        study_dir.mkdir(parents=True)
        with open(study_dir / "results.json", "w") as f:
            json.dump(mock_demographics[study_id], f)
        with open(study_dir / "info.json", "w") as f:
            json.dump({"date": "2025-04-19", "valid": True}, f)

    # Initialize extractor and pipeline info
    extractor = PatientStudyExtractor()
    input_pipeline_info = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": Path(demographics_dir),
        }
    }

    # First transformation with reduced dataset
    output_dir = tmp_path / "output"
    extractor.transform_dataset(
        reduced_dataset, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Add removed study back to demographics
    removed_study_dir = config_dir / removed_study_id
    removed_study_dir.mkdir(parents=True)
    with open(removed_study_dir / "results.json", "w") as f:
        json.dump(mock_demographics[removed_study_id], f)
    with open(removed_study_dir / "info.json", "w") as f:
        json.dump({"date": "2025-04-19", "valid": True}, f)

    # Second transformation with full dataset
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Verify added study results
    output_version_dir = output_dir / "PatientStudyExtractor" / extractor._version
    hash_dir = next(output_version_dir.iterdir())
    assert (hash_dir / removed_study_id / "results.json").exists()


def test_demographics_update(sample_data, mock_demographics, tmp_path):
    """Test that transform_dataset correctly reflects demographics updates."""
    # Setup initial demographics data
    demographics_dir = tmp_path / "participant_demographics"
    version_dir = demographics_dir / "1.0.0"
    config_dir = version_dir / "abc123"
    config_dir.mkdir(parents=True)

    with open(config_dir / "pipeline_info.json", "w") as f:
        pipeline_info = {
            "date": "2025-04-19",
            "version": "1.0.0",
            "config_hash": "abc123",
            "extractor": "DemographicsExtractor",
            "extractor_kwargs": {},
            "transform_kwargs": {},
            "input_pipelines": {},
            "schema": {},
        }
        json.dump(pipeline_info, f)

    # Write initial studies
    study_ids = list(mock_demographics.keys())
    test_study_id = study_ids[0]  # Use first study for testing updates

    for study_id, results in mock_demographics.items():
        study_dir = config_dir / study_id
        study_dir.mkdir(parents=True)
        with open(study_dir / "results.json", "w") as f:
            json.dump(results, f)
        with open(study_dir / "info.json", "w") as f:
            json.dump({"date": "2025-04-19", "valid": True}, f)

    # Initialize extractor and pipeline info
    extractor = PatientStudyExtractor()
    input_pipeline_info = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": Path(demographics_dir),
        }
    }

    # First transformation
    output_dir = tmp_path / "output"
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Update demographics for test study
    updated_demographics = {
        "groups": [
            {
                "name": "healthy",
                "age_mean": 45.0,
            },  # Changed from "patient" to "healthy"
            {"name": "control", "age_mean": 44.0},
        ]
    }
    with open(config_dir / test_study_id / "results.json", "w") as f:
        json.dump(updated_demographics, f)

    # Second transformation
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Verify updated results
    output_version_dir = output_dir / "PatientStudyExtractor" / extractor._version
    hash_dir = next(output_version_dir.iterdir())
    updated_results = json.loads(
        (hash_dir / test_study_id / "results.json").read_text()
    )
    assert updated_results["patient_study"] is False


def test_text_and_demographics_update(sample_data, mock_demographics, tmp_path):
    """Test text content and demographics updates are reflected correctly."""
    # Setup initial demographics data
    demographics_dir = tmp_path / "participant_demographics"
    version_dir = demographics_dir / "1.0.0"
    config_dir = version_dir / "abc123"
    config_dir.mkdir(parents=True)

    with open(config_dir / "pipeline_info.json", "w") as f:
        pipeline_info = {
            "date": "2025-04-19",
            "version": "1.0.0",
            "config_hash": "abc123",
            "extractor": "DemographicsExtractor",
            "extractor_kwargs": {},
            "transform_kwargs": {},
            "input_pipelines": {},
            "schema": {},
        }
        json.dump(pipeline_info, f)

    # Write initial studies
    study_ids = list(mock_demographics.keys())
    test_study_id = study_ids[0]  # Use first study for testing updates

    for study_id, results in mock_demographics.items():
        study_dir = config_dir / study_id
        study_dir.mkdir(parents=True)
        with open(study_dir / "results.json", "w") as f:
            json.dump(results, f)
        with open(study_dir / "info.json", "w") as f:
            json.dump({"date": "2025-04-19", "valid": True}, f)

    # Initialize extractor and pipeline info
    extractor = PatientStudyExtractor()
    input_pipeline_info = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": Path(demographics_dir),
        }
    }

    # First transformation
    output_dir = tmp_path / "output"
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Get first run info.json for comparison
    output_version_dir = output_dir / "PatientStudyExtractor" / extractor._version
    hash_dir = next(output_version_dir.iterdir())
    first_run_info = json.loads((hash_dir / test_study_id / "info.json").read_text())

    # Create modified dataset with updated text content
    modified_dataset = sample_data.slice([test_study_id])
    modified_dataset.data[test_study_id].pubget.text = Path(
        tmp_path / "modified_text.txt"
    )
    with open(modified_dataset.data[test_study_id].pubget.text, "w") as f:
        f.write("Modified text content with new patient information")

    # Update demographics for test study
    updated_demographics = {
        "groups": [
            {"name": "patient", "age_mean": 46.0},  # Updated age
            {"name": "control", "age_mean": 45.0},
            {"name": "experimental", "age_mean": 44.0},  # Added new group
        ]
    }
    with open(config_dir / test_study_id / "results.json", "w") as f:
        json.dump(updated_demographics, f)

    # Second transformation with modified dataset and updated demographics
    transform_kwargs = {
        "output_directory": output_dir,
        "input_pipeline_info": input_pipeline_info,
    }
    extractor.transform_dataset(modified_dataset, **transform_kwargs)

    # Get second run info.json
    hash_dir = next(output_version_dir.iterdir())
    second_run_info = json.loads((hash_dir / test_study_id / "info.json").read_text())

    # Verify input file changes are tracked in info.json
    assert (
        first_run_info["inputs"] != second_run_info["inputs"]
    ), "Expected input file hashes to change after modifying text content"
    assert (
        len(second_run_info["inputs"]) == 2
    )  # Should have text and demographics files
    assert any(
        "modified_text.txt" in file_path
        for file_path in second_run_info["inputs"].keys()
    ), "New text file path should appear in inputs"

    # Verify updated results
    updated_results = json.loads(
        (hash_dir / test_study_id / "results.json").read_text()
    )
    assert updated_results["patient_study"] is True
