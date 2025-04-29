import json
import random
from pathlib import Path
import pytest

from ns_extract.dataset import Dataset
from ns_extract.pipelines.umls_disease.model import UMLSDiseaseExtractor
from ns_extract.pipelines.participant_demographics.schemas import (
    BaseDemographicsSchema,
    GroupImaging,
)


@pytest.fixture
def sample_data(request) -> Dataset:
    """Load the sample dataset"""
    sample_path = Path("tests/data/sample_inputs")
    return Dataset(sample_path)


@pytest.fixture
def mock_demographics(sample_data) -> dict:
    """Generate mock demographics data using schema and real study IDs"""
    # Common diagnoses that should match UMLS
    diagnoses = [
        "Alzheimer's disease",
        "major depressive disorder",
        "schizophrenia",
        "bipolar disorder",
        "anxiety disorder",
        "PTSD",
        "ADHD",
        "autism spectrum disorder",
        "multiple sclerosis",
        "epilepsy",
    ]

    results = {}
    for study_id, study in sample_data.data.items():
        # Get study text to find positions
        text = None
        for source in ["pubget", "ace"]:
            source_obj = getattr(study, source, None)
            if source_obj and source_obj.text:
                with open(source_obj.text) as f:
                    text = f.read()
                break

        if not text:
            continue

        # Generate 1-3 groups
        groups = []
        n_groups = random.randint(1, 3)

        for i in range(n_groups):
            # Generate a group with proper schema
            total = random.randint(20, 100)
            male_count = random.randint(0, total)
            female_count = total - male_count

            group = {
                "count": total,
                "diagnosis": random.choice(diagnoses),
                "group_name": "patients",  # All groups with diagnoses are patients
                "subgroup_name": f"Group {i+1}",
                "male_count": male_count,
                "female_count": female_count,
                "age_mean": round(random.uniform(18, 65), 1),
                "age_range": f"{random.randint(18, 30)}-{random.randint(55, 80)}",
                "age_minimum": random.randint(18, 30),
                "age_maximum": random.randint(55, 80),
                "age_median": random.randint(35, 50),
                "imaging_sample": random.choice(["yes", "no"]),
            }

            # Validate with proper schema
            validated_group = GroupImaging(**group).model_dump()
            groups.append(validated_group)

        # Create results for this study
        results[study_id] = BaseDemographicsSchema(groups=groups).model_dump()

    return results


def test_umls_disease_extractor(sample_data, mock_demographics, tmp_path):
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
    extractor = UMLSDiseaseExtractor(
        inputs=("text",),
        input_sources=("pubget", "ace"),
        pipeline_inputs={"participant_demographics": ["results"]},
    )

    # Set up pipeline kwargs
    pipeline_kwargs = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_directory": demographics_dir,
        }
    }

    # Run extraction
    output_dir = tmp_path / "output"
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_kwargs=pipeline_kwargs
    )

    # Verify outputs
    result_dir = output_dir / "UMLSDiseaseExtractor" / "1.0.0"
    assert result_dir.exists()

    # Get the hash directory
    hash_dir = next(result_dir.glob("*"))

    # Check results for each study
    for study_id, demographics in mock_demographics.items():
        study_dir = hash_dir / study_id
        if not study_dir.exists():
            continue

        results_file = study_dir / "results.json"
        assert results_file.exists()

        with open(results_file) as f:
            results = json.load(f)

        # Each diagnosis in demographics should have UMLS entities
        for group in demographics["groups"]:
            matching_results = [
                r for r in results if r["diagnosis"] == group["diagnosis"]
            ]

            if matching_results:
                result = matching_results[0]
                assert result["count"] == group["count"]
                assert len(result["umls_entities"]) >= 1


def test_missing_demographics_pipeline(sample_data, mock_demographics, tmp_path):
    # Initialize extractor
    extractor = UMLSDiseaseExtractor(
        inputs=("text",),
        input_sources=("pubget", "ace"),
        pipeline_inputs={"participant_demographics": ["results"]},
    )

    # Set up pipeline kwargs with wrong directory
    pipeline_kwargs = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_directory": tmp_path / "wrong_dir",
        }
    }

    # Run extraction - should raise error
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match=".*No version directories found.*"):
        extractor.transform_dataset(
            sample_data, output_dir, input_pipeline_kwargs=pipeline_kwargs
        )


def test_missing_demographics_results(sample_data, mock_demographics, tmp_path):
    # Create minimal pipeline structure
    config_dir = tmp_path / "participant_demographics/1.0.0/abc123"
    config_dir.mkdir(parents=True)

    # Write pipeline info only
    with open(config_dir / "pipeline_info.json", "w") as f:
        json.dump(
            {
                "date": "2025-04-19",
                "version": "1.0.0",
                "type": "participant_demographics",
            },
            f,
        )

    # Initialize extractor
    extractor = UMLSDiseaseExtractor(
        inputs=("text",),
        input_sources=("pubget", "ace"),
        pipeline_inputs={"participant_demographics": ["results"]},
    )

    # Set up pipeline kwargs
    pipeline_kwargs = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_directory": tmp_path / "participant_demographics",
        }
    }

    # Run extraction - should raise error for missing results
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match=".*Missing results.json.*"):
        extractor.transform_dataset(
            sample_data, output_dir, input_pipeline_kwargs=pipeline_kwargs
        )
