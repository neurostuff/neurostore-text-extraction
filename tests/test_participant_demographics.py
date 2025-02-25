import pytest
from pathlib import Path

from ns_pipelines import ParticipantDemographicsExtractor
from ns_pipelines.dataset import Dataset


@pytest.mark.vcr(record_mode="once")
def test_ParticipantDemographicsExtractor(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    pde = ParticipantDemographicsExtractor(
        extraction_model="gpt-4o-mini-2024-07-18",
        env_variable="OPENAI_API_KEY",
    )
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "participant_demographics"
    pde.transform_dataset(dataset, output_dir)
    assert True
