"""Tests for the CLI module."""

import pytest
import yaml
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch

from ns_extract.cli.run import main, get_pipeline_map, load_yaml_config
from ns_extract.dataset import Dataset
from ns_extract.pipelines import (
    WordCountExtractor,
    ParticipantDemographicsExtractor,
    TaskExtractor,
)


def run_cli(args):
    """Helper function to run CLI with arguments and capture output."""
    stdout = StringIO()
    stderr = StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr), patch(
        "sys.argv", ["ns_extract"] + args
    ):
        try:
            main()
            return 0, stdout.getvalue(), stderr.getvalue()
        except SystemExit as e:
            return e.code, stdout.getvalue(), stderr.getvalue()


def create_test_config(tmp_path, config_data):
    """Create a test pipeline config file."""
    config_path = tmp_path / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def output_path(tmp_path):
    """Create test environment with dataset and output directory."""
    output_path = tmp_path / "output"
    return output_path


def test_cli_with_pipeline_args(output_path, sample_data):
    """Test CLI with direct pipeline arguments."""

    exit_code, stdout, stderr = run_cli(
        [str(sample_data), str(output_path), "--pipelines", "word_count"]
    )

    assert exit_code == 0
    assert "Running word_count pipeline" in stdout
    assert "Completed word_count pipeline" in stdout

    # Check output directory was created
    assert (output_path / "word_count").exists()


def test_cli_with_yaml_config(output_path, sample_data):
    """Test CLI with YAML configuration file."""
    # Create test config
    config = {"pipelines": [{"name": "word_count", "args": {"square_root": True}}]}
    config_path = create_test_config(sample_data.parent, config)

    exit_code, stdout, stderr = run_cli(
        [str(sample_data), str(output_path), "--config", str(config_path)]
    )

    assert exit_code == 0
    assert "Running word_count pipeline" in stdout
    assert "Completed word_count pipeline" in stdout

    # Check output directory was created
    assert (output_path / "word_count").exists()


def test_cli_invalid_pipeline(output_path, sample_data):
    """Test CLI with invalid pipeline name."""
    exit_code, stdout, stderr = run_cli(
        [str(sample_data), str(output_path), "--pipelines", "invalid_pipeline"]
    )

    assert exit_code != 0
    assert "invalid_pipeline" in stderr


def test_cli_invalid_yaml(output_path, sample_data):
    """Test CLI with invalid YAML configuration."""
    # Create invalid config
    config = {"invalid": "config"}
    config_path = create_test_config(sample_data.parent, config)

    exit_code, stdout, stderr = run_cli(
        [str(sample_data), str(output_path), "--config", str(config_path)]
    )

    assert exit_code != 0
    assert "YAML file must contain a 'pipelines' list" in stderr


def test_cli_no_pipeline_specified(output_path, sample_data):
    """Test CLI with no pipeline specified."""
    exit_code, stdout, stderr = run_cli([str(sample_data), str(output_path)])

    assert exit_code != 0
    assert "error: one of the arguments --pipelines --config is required" in stderr


def test_cli_both_methods_specified(output_path, sample_data):
    """Test CLI with both --pipelines and --config specified."""
    config = {"pipelines": ["word_count"]}
    config_path = create_test_config(sample_data.parent, config)

    exit_code, stdout, stderr = run_cli(
        [
            str(sample_data),
            str(output_path),
            "--pipelines",
            "word_count",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code != 0
    assert "not allowed with argument" in stderr


def test_cli_pipeline_error(output_path, sample_data, monkeypatch):
    """Test CLI handles pipeline execution errors."""

    # Mock transform_dataset to raise an error
    def mock_transform(*args, **kwargs):
        raise ValueError("Pipeline error")

    monkeypatch.setattr(WordCountExtractor, "transform_dataset", mock_transform)

    exit_code, stdout, stderr = run_cli(
        [str(sample_data), str(output_path), "--pipelines", "word_count"]
    )

    assert exit_code != 0
    assert "Error running word_count pipeline" in stderr
    assert "Pipeline error" in stderr


def test_get_pipeline_map():
    """Test pipeline name mapping function."""
    pipeline_map = get_pipeline_map()

    # Test CamelCase to snake_case conversion
    assert "word_count" in pipeline_map
    assert "participant_demographics" in pipeline_map
    assert "task" in pipeline_map

    # Test correct class mapping
    assert pipeline_map["word_count"] == WordCountExtractor
    assert pipeline_map["participant_demographics"] == ParticipantDemographicsExtractor
    assert pipeline_map["task"] == TaskExtractor


def test_load_yaml_config_simple_format(tmp_path):
    """Test loading YAML config with simple pipeline names."""
    config = {"pipelines": ["word_count", "task"]}
    config_path = create_test_config(tmp_path, config)

    available_pipelines = {"word_count", "task", "participant_demographics"}
    pipeline_configs = load_yaml_config(config_path, available_pipelines)

    assert len(pipeline_configs) == 2
    assert pipeline_configs[0] == ("word_count", {})
    assert pipeline_configs[1] == ("task", {})


def test_load_yaml_config_multiple_pipelines(output_path, tmp_path):
    """Test loading YAML config with multiple pipelines and arguments."""
    config = {
        "pipelines": [
            {"name": "word_count", "args": {"min_word_length": 3}},
            {"name": "task", "args": {"model_name": "gpt-4"}},
            "participant_demographics",  # Simple format mixed with detailed format
        ]
    }
    config_path = create_test_config(tmp_path, config)

    available_pipelines = {"word_count", "task", "participant_demographics"}
    pipeline_configs = load_yaml_config(config_path, available_pipelines)

    assert len(pipeline_configs) == 3
    assert pipeline_configs[0] == ("word_count", {"min_word_length": 3})
    assert pipeline_configs[1] == ("task", {"model_name": "gpt-4"})
    assert pipeline_configs[2] == ("participant_demographics", {})


def test_load_yaml_config_invalid_format(tmp_path):
    """Test loading YAML config with invalid pipeline format."""
    config = {"pipelines": [{"invalid": "format"}]}
    config_path = create_test_config(tmp_path, config)

    available_pipelines = {"word_count", "task"}

    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        load_yaml_config(config_path, available_pipelines)


def test_load_yaml_config_file_error(tmp_path):
    """Test error handling for unreadable YAML file."""
    config_path = tmp_path / "nonexistent.yaml"

    available_pipelines = {"word_count"}

    with pytest.raises(FileNotFoundError):
        load_yaml_config(config_path, available_pipelines)


def test_load_yaml_config_invalid_yaml(tmp_path):
    """Test error handling for invalid YAML content."""
    config_path = tmp_path / "invalid.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: content:\nthis is not valid yaml")

    available_pipelines = {"word_count"}

    with pytest.raises(ValueError, match="Error parsing YAML file"):
        load_yaml_config(config_path, available_pipelines)
