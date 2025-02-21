import pytest
from pathlib import Path

from ns_pipelines import TaskExtractor
from ns_pipelines.dataset import Dataset


@pytest.mark.vcr(record_mode="once", filter_headers=["authorization"])
def test_TaskExtractor(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    pde = TaskExtractor(
        extraction_model="gpt-4o-mini-2024-07-18",
        prompt_set="ZERO_SHOT_TASK",
        env_variable="API_CLIENT_OPENAI_KEY",
        env_file=str(Path(__file__).parents[1] / ".keys"),
    )
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "participant_demographics"
    pde.run(dataset, output_dir)
    assert True
