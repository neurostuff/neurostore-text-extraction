import json
import pytest

from ns_extract.pipelines import ParticipantDemographicsExtractor
from ns_extract.dataset import Dataset
from ns_extract.pipelines.participant_demographics.schemas import (
    BaseDemographicsSchema,
    GroupImaging,
)


@pytest.mark.vcr(record_mode="once", filter_headers=["authorization"])
def test_ParticipantDemographicsExtractor(sample_data, tmp_path):
    """Test the participant demographics extraction pipeline."""
    # Initialize extractor
    pde = ParticipantDemographicsExtractor(
        extraction_model="gpt-4o-mini-2024-07-18",
        env_variable="OPENAI_API_KEY",
    )
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "participant_demographics"

    # Initial run
    pde.transform_dataset(dataset, output_dir)

    # Verify directory structure and files
    version_dir = next(output_dir.glob("ParticipantDemographicsExtractor/1.0.0/*"))
    assert version_dir.exists()

    # Check pipeline info
    pipeline_info = json.loads((version_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == "1.0.0"
    assert (
        pipeline_info["extractor_kwargs"]["extraction_model"]
        == "gpt-4o-mini-2024-07-18"
    )

    # Verify study outputs and schema validation
    for study_dir in version_dir.glob("*"):
        if study_dir.is_dir():
            results_file = study_dir / "results.json"
            info_file = study_dir / "info.json"
            assert results_file.exists()
            assert info_file.exists()

            info = json.loads(info_file.read_text())
            # date, input, and valid is required
            assert info["date"]
            # assert info["valid"] == True

            # Load and validate results
            results = json.loads(results_file.read_text())

            # Validate against BaseDemographicsSchema
            validated = BaseDemographicsSchema.model_validate(results)

            # Check groups
            assert validated.groups
            for group in validated.groups:
                assert isinstance(group, GroupImaging)

                # Validate required group fields
                assert isinstance(group.count, int)
                assert group.count >= 0

                assert group.group_name in ["healthy", "patients"]

                assert isinstance(group.male_count, int) or group.male_count is None
                if group.male_count:
                    assert group.male_count >= 0

                assert isinstance(group.female_count, int) or group.female_count is None
                if group.female_count:
                    assert group.female_count >= 0

                # Validate age fields
                assert isinstance(group.age_mean, float) or group.age_mean is None
                if group.age_mean:
                    assert group.age_mean > 0

                assert isinstance(group.age_minimum, int) or group.age_minimum is None
                assert isinstance(group.age_maximum, int) or group.age_maximum is None
                assert isinstance(group.age_median, int) or group.age_median is None

                # Validate imaging sample
                assert group.imaging_sample in ["yes", "no"]
