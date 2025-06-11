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
                {"name": f"group{i}", "age_mean": 45.0 + i},
                {"name": "control", "age_mean": 44.0 + i},
            ]
        }
    return mock_data


def test_text_normalization_and_expansion(sample_data, mock_demographics, tmp_path):
    """Test that text is properly normalized and abbreviations are expanded."""
    # Setup demographics
    demographics_dir = setup_demographics_dir(tmp_path, mock_demographics)

    # Create test dataset with text containing abbreviations and mixed case
    test_study_id = list(mock_demographics.keys())[0]
    modified_dataset = sample_data.slice([test_study_id])
    modified_dataset.data[test_study_id].pubget.text = Path(tmp_path / "test_text.txt")
    # Prepare test text with mixed case and abbreviations
    test_text = (
        "TEST with Magnetic Resonance Imaging (MRI) and "
        "Electroencephalogram (EEG) DATA"
    )
    with open(modified_dataset.data[test_study_id].pubget.text, "w") as f:
        f.write(test_text)

    # Initialize extractor
    extractor = ExampleExtractor()
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
        modified_dataset, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Check results
    output_version_dir = output_dir / "ExampleExtractor" / extractor._version
    hash_dir = next(output_version_dir.iterdir())
    results = json.loads((hash_dir / test_study_id / "results.json").read_text())

    # Verify text is normalized and abbreviations are expanded
    result_value = results["value"]

    # Original text had "TEST" - verify case normalization to title case
    assert (
        "test" in result_value and "TEST" not in result_value
    ), "Text should be normalized to Cap case"

    # Original text had "Magnetic Resonance Imaging (MRI)" - verify both forms present
    assert (
        "Magnetic Resonance Imaging" in result_value
    ), "Long form 'Magnetic Resonance Imaging' should be present"

    # Original text had "Electroencephalogram (EEG)" - verify both forms present
    assert (
        "Electroencephalogram" in result_value
    ), "Long form 'Electroencephalogram' should be present"


def setup_demographics_dir(tmp_path, mock_demographics):
    """Helper to set up demographics pipeline directory."""
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

    return demographics_dir


def test_disabled_abbreviation_expansion(sample_data, mock_demographics, tmp_path):
    """Test that abbreviation expansion can be disabled while keeping normalization."""
    # Setup demographics
    demographics_dir = setup_demographics_dir(tmp_path, mock_demographics)

    # Create test dataset with text containing abbreviations and mixed case
    test_study_id = list(mock_demographics.keys())[0]
    modified_dataset = sample_data.slice([test_study_id])
    modified_dataset.data[test_study_id].pubget.text = Path(tmp_path / "test_text.txt")

    # Prepare test text with mixed case and abbreviations
    test_text = (
        "TEST with Magnetic Resonance Imaging (MRI) and "
        "Electroencephalogram (EEG) DATA"
    )
    with open(modified_dataset.data[test_study_id].pubget.text, "w") as f:
        f.write(test_text)

    # Initialize extractor with abbreviation expansion disabled
    extractor = ExampleExtractor(disable_abbreviation_expansion=True)
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
        modified_dataset, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Check results
    output_version_dir = output_dir / "ExampleExtractor" / extractor._version
    hash_dir = next(output_version_dir.iterdir())
    results = json.loads((hash_dir / test_study_id / "results.json").read_text())

    result_value = results["value"]

    # Text should still be normalized to Cap case
    assert (
        "test" in result_value and "TEST" not in result_value
    ), "Text should still be normalized even when abbreviation expansion is disabled"

    # Abbreviations should not be expanded
    assert (
        "Magnetic Resonance Imaging" in result_value
    ), "Original abbreviation form should be preserved when expansion is disabled"

    assert (
        "Electroencephalogram" in result_value
    ), "Original abbreviation form should be preserved when expansion is disabled"


def test_idempotency(sample_data, mock_demographics, tmp_path):
    """Test that running transform_dataset twice produces identical results."""
    demographics_dir = setup_demographics_dir(tmp_path, mock_demographics)

    extractor = ExampleExtractor(disable_abbreviation_expansion=True)
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
    output_version_dir = output_dir / "ExampleExtractor" / extractor._version
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


def test_remove_and_read_study(sample_data, mock_demographics, tmp_path):
    """Test that removing and re-adding a study works correctly."""
    demographics_dir = setup_demographics_dir(tmp_path, mock_demographics)

    # Get list of study IDs
    study_ids = list(mock_demographics.keys())
    removed_study_id = study_ids[0]  # Take first study to remove and add back
    remaining_study_ids = study_ids[1:]

    # Create reduced dataset without the first study
    reduced_dataset = sample_data.slice(remaining_study_ids)

    extractor = ExampleExtractor(disable_abbreviation_expansion=True)
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

    # Second transformation with full dataset
    hash_dir = extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Check results.json exists and contains expected fields
    results = json.loads((hash_dir / removed_study_id / "results.json").read_text())
    assert "value" in results
    assert "confidence" in results
    assert isinstance(results["confidence"], float)
    assert 0 <= results["confidence"] <= 1


def test_demographics_update(sample_data, mock_demographics, tmp_path):
    """Test that demographics updates are reflected correctly."""
    demographics_dir = setup_demographics_dir(tmp_path, mock_demographics)

    study_ids = list(mock_demographics.keys())
    test_study_id = study_ids[0]

    extractor = ExampleExtractor(disable_abbreviation_expansion=True)
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

    # Get first run info and hash directory
    output_version_dir = output_dir / "ExampleExtractor" / extractor._version
    first_hash_dir = next(output_version_dir.iterdir())
    first_run_info = json.loads(
        (first_hash_dir / test_study_id / "info.json").read_text()
    )

    # Store first hash directory path for comparison
    first_hash_path = first_hash_dir

    # Update demographics for test study
    updated_demographics = {
        "groups": [
            {"name": "updated_group", "age_mean": 50.0},
            {"name": "control", "age_mean": 49.0},
        ]
    }

    config_dir = demographics_dir / "1.0.0" / "abc123"
    with open(config_dir / test_study_id / "results.json", "w") as f:
        json.dump(updated_demographics, f)

    # Second transformation
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Get second run hash directory
    second_hash_dir = next(
        d for d in output_version_dir.iterdir() if d != first_hash_path
    )
    second_run_info = json.loads(
        (second_hash_dir / test_study_id / "info.json").read_text()
    )

    # Verify info.json differences and separate hash directories
    assert first_run_info != second_run_info
    assert first_hash_dir != second_hash_dir


def test_text_and_demographics_update(sample_data, mock_demographics, tmp_path):
    """Test both text content and demographics updates are reflected correctly."""
    demographics_dir = setup_demographics_dir(tmp_path, mock_demographics)

    study_ids = list(mock_demographics.keys())
    test_study_id = study_ids[0]

    extractor = ExampleExtractor(disable_abbreviation_expansion=True)
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

    # Get first run info and hash directory
    output_version_dir = output_dir / "ExampleExtractor" / extractor._version
    first_hash_dir = next(output_version_dir.iterdir())
    first_run_info = json.loads(
        (first_hash_dir / test_study_id / "info.json").read_text()
    )

    # Store first hash directory path for comparison
    first_hash_path = first_hash_dir

    # Create modified dataset with updated text content
    modified_dataset = sample_data.slice([test_study_id])
    modified_dataset.data[test_study_id].pubget.text = Path(
        tmp_path / "modified_text.txt"
    )
    with open(modified_dataset.data[test_study_id].pubget.text, "w") as f:
        f.write("Modified example text content")

    # Update demographics
    updated_demographics = {
        "groups": [
            {"name": "updated_group", "age_mean": 50.0},
            {"name": "new_group", "age_mean": 48.0},
        ]
    }

    config_dir = demographics_dir / "1.0.0" / "abc123"
    with open(config_dir / test_study_id / "results.json", "w") as f:
        json.dump(updated_demographics, f)

    # Second transformation
    extractor.transform_dataset(
        modified_dataset, output_dir, input_pipeline_info=input_pipeline_info
    )

    # Get second run hash directory and info
    second_hash_dir = next(
        d for d in output_version_dir.iterdir() if d != first_hash_path
    )
    second_run_info = json.loads(
        (second_hash_dir / test_study_id / "info.json").read_text()
    )

    # Verify info.json differences and separate hash directories
    assert first_run_info != second_run_info
    assert first_hash_dir != second_hash_dir
    assert "modified_text.txt" in str(second_run_info["inputs"])


def test_post_process_and_file_handling(sample_data, mock_demographics, tmp_path):
    """Test post-processing modes and file handling behavior."""
    demographics_dir = setup_demographics_dir(tmp_path, mock_demographics)

    # Create test data with clear transformation differences
    test_study_id = list(mock_demographics.keys())[0]
    modified_dataset = sample_data.slice([test_study_id])
    modified_dataset.data[test_study_id].pubget.text = Path(tmp_path / "test_text.txt")
    test_text = "TEST with Magnetic Resonance Imaging (MRI) DATA"
    with open(modified_dataset.data[test_study_id].pubget.text, "w") as f:
        f.write(test_text)

    # Set group name in lowercase to test normalization
    mock_demographics[test_study_id]["groups"][0]["name"] = "test_group"
    input_pipeline_info = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": Path(demographics_dir),
        }
    }
    extractor = ExampleExtractor()

    # Test 1: Create initial results with post_process=False
    output_dir = tmp_path / "output"
    hash_dir = extractor.transform_dataset(
        modified_dataset,
        output_dir,
        post_process=False,
        overwrite=True,
        input_pipeline_info=input_pipeline_info,
    )

    # Get study path
    study_dir = hash_dir / test_study_id

    # Verify raw results weren't post-processed
    with open(study_dir / "results.json") as f:
        raw_results = json.load(f)
    assert not raw_results[
        "was_post_processed"
    ], "Raw results should not be post-processed"

    # Test 2: Post-process existing results with post_process="only"
    hash_dir2 = extractor.transform_dataset(
        modified_dataset,
        output_dir,
        post_process="only",
        overwrite=True,
        input_pipeline_info=input_pipeline_info,
    )
    assert hash_dir == hash_dir2, "Should use same directory"

    # Verify was_post_processed flag is set
    with open(study_dir / "results.json") as f:
        processed_results = json.load(f)
    assert processed_results[
        "was_post_processed"
    ], "Results should be marked as post-processed"

    # Test 3: Try post-processing again with overwrite=False
    hash_dir3 = extractor.transform_dataset(
        modified_dataset,
        output_dir,
        post_process=True,
        overwrite=False,
        input_pipeline_info=input_pipeline_info,
    )
    assert hash_dir3 == hash_dir, "Should use same directory for all operations"

    # Verify none of the files were modified
    with open(study_dir / "results.json") as f:
        final_results = json.load(f)

    assert (
        final_results == processed_results
    ), "Results shouldn't change with overwrite=False"
    assert final_results[
        "was_post_processed"
    ], "Post-processed state should be preserved"
