import json
import pytest
from .example_pipelines.word_count.model import (
    WordCountExtractor,
    WordDevianceExtractor,
)
from ns_extract.dataset import Dataset


@pytest.fixture
def text_content():
    """Sample text content for testing."""
    return "This is a test document with exactly ten words, wow."


@pytest.fixture
def mock_inputs(text_content):
    """Mock study inputs for testing."""
    return {
        "study1": {"text": text_content},
        "study2": {"text": " ".join([text_content, text_content])},  # Double length
    }


def test_word_count_transform(text_content):
    """Test WordCountExtractor.execute() with preprocessed inputs."""
    extractor = WordCountExtractor()
    cleaned_result, result, valid = extractor.transform({"dummy_study_id": {"text": text_content}})
    assert result['dummy_study_id']["word_count"] == 10
    assert valid['dummy_study_id'] is True


def test_word_deviance_transform(mock_inputs):
    """Test WordDevianceExtractor._transform() with preprocessed inputs."""
    extractor = WordDevianceExtractor()
    cleaned_result, results, valid = extractor.transform(mock_inputs)

    # First document has 10 words, second has 20
    # Average is 15, so deviances should be 5
    assert results["study1"]["word_deviance"] == 5
    assert results["study2"]["word_deviance"] == 5
    assert valid["study1"] is True
    assert valid["study2"] is True


def test_WordCountExtractor(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    wce = WordCountExtractor()
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_count"

    # Initial run
    wce.transform_dataset(dataset, output_dir)

    # Verify directory structure and files
    version_dir = next(output_dir.glob("WordCountExtractor/1.0.0/*"))
    assert version_dir.exists()

    # Check pipeline info
    pipeline_info = json.loads((version_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == "1.0.0"

    # Verify study outputs
    # Glob for dirs
    study_dirs = list([x for x in version_dir.glob("*") if x.is_dir()])
    expected_len = len([d for d in sample_data.iterdir() if d.is_dir()])
    assert len(study_dirs) == expected_len
    for study_dir in study_dirs:
        results_file = study_dir / "results.json"
        info_file = study_dir / "info.json"
        assert results_file.exists()
        assert info_file.exists()

        # Validate results schema
        results = json.loads(results_file.read_text())
        assert "word_count" in results
        assert isinstance(results["word_count"], int)

    # Rerun - no changes, no new outputs
    wce.transform_dataset(dataset, output_dir)
    assert len(list(output_dir.glob("WordCountExtractor/1.0.0/*"))) == 1


def test_parallel_processing(sample_data, tmp_path):
    """Test that parallel processing works correctly with WordCountExtractor."""
    # Create extractor
    wce = WordCountExtractor()
    dataset = Dataset(sample_data)

    # Run with different num_workers values
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    # Run in serial and parallel mode
    wce.transform_dataset(dataset, serial_dir, num_workers=1)
    wce.transform_dataset(dataset, parallel_dir, num_workers=4)

    # Get results from both runs
    serial_version_dir = next(serial_dir.glob("WordCountExtractor/1.0.0/*"))
    parallel_version_dir = next(parallel_dir.glob("WordCountExtractor/1.0.0/*"))

    # Verify same number of studies processed
    serial_studies = list(serial_version_dir.glob("*/results.json"))
    parallel_studies = list(parallel_version_dir.glob("*/results.json"))
    assert len(serial_studies) == len(parallel_studies)

    # Compare results
    for serial_result in serial_studies:
        # Find matching parallel result
        study_id = serial_result.parent.name
        parallel_result = parallel_version_dir / study_id / "results.json"

        assert parallel_result.exists()

        # Compare contents
        serial_data = json.loads(serial_result.read_text())
        parallel_data = json.loads(parallel_result.read_text())
        assert serial_data == parallel_data


def test_WordDevianceExtractor(sample_data, tmp_path):
    """Test the word deviance extraction pipeline."""
    wde = WordDevianceExtractor()
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_deviance"

    # Initial run
    wde.transform_dataset(dataset, output_dir)

    # Verify directory structure and files
    version_dir = next(output_dir.glob("WordDevianceExtractor/1.0.0/*"))
    assert version_dir.exists()

    # Check pipeline info
    pipeline_info = json.loads((version_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == "1.0.0"

    # Verify study outputs
    study_dirs = list([x for x in version_dir.glob("*") if x.is_dir()])
    expected_len = len([d for d in sample_data.iterdir() if d.is_dir()])
    assert len(study_dirs) == expected_len
    for study_dir in study_dirs:
        results_file = study_dir / "results.json"
        info_file = study_dir / "info.json"
        assert results_file.exists()
        assert info_file.exists()

        # Validate results schema
        results = json.loads(results_file.read_text())
        assert "word_deviance" in results
        assert isinstance(results["word_deviance"], int)

    assert len(list(output_dir.glob("WordDevianceExtractor/1.0.0/*"))) == 1

    # Rerun - no changes, no new outputs
    wde.transform_dataset(dataset, output_dir)

    # No new directory since no changes in inputs
    assert len(list(output_dir.glob("WordDevianceExtractor/1.0.0/*"))) == 1
