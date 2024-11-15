import pytest

from ns_pipelines.participant_demographics.run import ParticipantDemographicsExtraction
from ns_pipelines.dataset import Dataset


@pytest.mark.vcr(record_mode="once", filter_headers=["authorization"])
def test_ParticipantDemographicsExtraction(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    pde = ParticipantDemographicsExtraction(
        extraction_model="gpt-4o-2024-08-06",
        prompt_set="ZERO_SHOT_MULTI_GROUP_FTSTRICT_FC"
    )
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "participant_demographics"
    pde.run(dataset, output_dir)
    assert True
