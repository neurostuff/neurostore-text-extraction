import json
import random
from pathlib import Path
import pytest
import spacy

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


@pytest.mark.skip(reason="UMLS tests are optional")
def test_umls_disease_transform_default_model(sample_data, mock_demographics):
    """Test UMLSDiseaseExtractor._transform() with real study data."""
    # Get studies with text data
    valid_studies = {
        sid: study
        for sid, study in sample_data.data.items()
        if sid in mock_demographics
        and ((study.pubget and study.pubget.text) or (study.ace and study.ace.text))
    }

    # We need at least one valid study
    assert len(valid_studies) > 0, "No valid studies found for testing"

    # Create test inputs for multiple studies
    inputs = {}
    for study_id, study in valid_studies.items():
        # Get study text
        text = None
        for source in ["pubget", "ace"]:
            source_obj = getattr(study, source, None)
            if source_obj and source_obj.text:
                with open(source_obj.text) as f:
                    text = f.read()
                break

        assert text is not None, f"No text found for study {study_id}"

        # Add study inputs
        inputs[study_id] = {
            "text": text,
            "participant_demographics": mock_demographics[study_id],
        }

    # Run extraction
    extractor = UMLSDiseaseExtractor()
    results = extractor._transform(inputs)

    # Verify structure and content for each study
    for study_id, study_results in results.items():
        # Should have groups list
        assert "groups" in study_results
        assert len(study_results["groups"]) > 0

        # Each group should have required fields and UMLS entities
        for result in study_results["groups"]:
            assert "diagnosis" in result
            assert "umls_entities" in result
            assert len(result["umls_entities"]) > 0
            assert "umls_cui" in result["umls_entities"][0]
            assert "umls_name" in result["umls_entities"][0]
            assert "umls_prob" in result["umls_entities"][0]

            # Verify group index and count match demographics
            demo_groups = mock_demographics[study_id]["groups"]
            assert result["group_ix"] < len(demo_groups)
            assert result["count"] == demo_groups[result["group_ix"]]["count"]


@pytest.mark.skip(reason="UMLS tests are optional")
def test_custom_model_configuration():
    """Test UMLSDiseaseExtractor with custom model configuration."""
    # Initialize extractor with a different model
    extractor = UMLSDiseaseExtractor(model_name="en_core_web_sm")
    assert extractor.model_name == "en_core_web_sm"
    assert "abbreviation_detector" in extractor.nlp.pipe_names
    assert "serialize_abbreviation" in extractor.nlp.pipe_names

    # Verify disabled components
    assert "parser" not in extractor.nlp.pipe_names
    assert "ner" not in extractor.nlp.pipe_names


@pytest.mark.skip(reason="UMLS tests are optional")
def test_invalid_model_name():
    """Test error handling for invalid model names."""
    with pytest.raises(SystemExit):
        UMLSDiseaseExtractor(model_name="invalid_model")


@pytest.mark.skip(reason="UMLS tests are optional")
def test_transformer_model_rejection():
    """Test rejection of transformer-based models."""
    with pytest.raises(ImportError, match=".*is a transformer model.*"):
        # Using a transformer model should raise an error
        UMLSDiseaseExtractor(model_name="en_core_web_trf")


@pytest.mark.skip(reason="UMLS tests are optional")
def test_umls_disease_extractor(sample_data, mock_demographics, tmp_path):
    """Test full pipeline extraction with multiple studies."""
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
    study_ids_with_text = []
    for study_id, results in mock_demographics.items():
        study = sample_data.data[study_id]
        # Only process studies with text data
        if not ((study.pubget and study.pubget.text) or (study.ace and study.ace.text)):
            continue

        study_dir = config_dir / study_id
        study_dir.mkdir(parents=True)

        # Write results and info files
        with open(study_dir / "results.json", "w") as f:
            json.dump(results, f)

        with open(study_dir / "info.json", "w") as f:
            json.dump({"date": "2025-04-19", "valid": True}, f)

        study_ids_with_text.append(study_id)

    assert len(study_ids_with_text) > 0, "No valid studies found for testing"

    # Initialize extractor
    extractor = UMLSDiseaseExtractor()

    # Set up pipeline kwargs
    pipeline_kwargs = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": demographics_dir,
        }
    }

    # Run extraction
    output_dir = tmp_path / "output"
    extractor.transform_dataset(
        sample_data, output_dir, input_pipeline_info=pipeline_kwargs
    )

    # Verify outputs
    result_dir = output_dir / "UMLSDiseaseExtractor" / "1.0.0"
    assert result_dir.exists()

    # Get the hash directory
    hash_dir = next(result_dir.glob("*"))

    # Track processed studies to ensure we test multiple
    processed_studies = 0

    # Check results for each study
    for study_id in study_ids_with_text:
        study_dir = hash_dir / study_id
        assert study_dir.exists(), f"Missing results for study {study_id}"

        results_file = study_dir / "results.json"
        assert results_file.exists()

        with open(results_file) as f:
            results = json.load(f)

        demo_groups = mock_demographics[study_id]["groups"]

        # Results should follow the new structure
        assert isinstance(results, dict)
        assert "groups" in results
        assert isinstance(results["groups"], list)

        # Verify groups match expected structure and correspond to demographics groups
        result_groups = results["groups"]
        for result_group in result_groups:
            # Basic structure checks
            assert isinstance(result_group, dict)
            assert "diagnosis" in result_group
            assert "umls_entities" in result_group
            assert "group_ix" in result_group
            assert "count" in result_group
            
            # Verify group matches demographics
            group_ix = result_group["group_ix"]
            assert group_ix < len(demo_groups)
            demo_group = demo_groups[group_ix]
            
            # Count should match
            assert result_group["count"] == demo_group["count"]
            
            # UMLS entities should be valid
            assert len(result_group["umls_entities"]) >= 1
            for entity in result_group["umls_entities"]:
                assert "umls_cui" in entity
                assert "umls_name" in entity
                assert "umls_prob" in entity
                assert isinstance(entity["umls_prob"], float)
                assert 0 <= entity["umls_prob"] <= 1
        
        processed_studies += 1

    # Ensure we tested multiple studies
    assert processed_studies > 1, "Not enough studies were processed"


@pytest.mark.skip(reason="UMLS tests are optional")
def test_missing_demographics_pipeline(sample_data, mock_demographics, tmp_path):
    # Initialize extractor
    extractor = UMLSDiseaseExtractor()

    # Set up pipeline kwargs with wrong directory
    pipeline_kwargs = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": tmp_path / "wrong_dir",
        }
    }

    # Run extraction - should raise error
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match=".*No version directories found.*"):
        extractor.transform_dataset(
            sample_data, output_dir, input_pipeline_info=pipeline_kwargs
        )


@pytest.mark.skip(reason="UMLS tests are optional")
def test_invalid_demographics_input(sample_data, mock_demographics):
    """Test handling of invalid demographics input data."""
    # Get a valid study to use as base
    study_id = next(
        sid
        for sid, study in sample_data.data.items()
        if sid in mock_demographics
        and ((study.pubget and study.pubget.text) or (study.ace and study.ace.text))
    )
    study = sample_data.data[study_id]

    # Get study text
    text = None
    for source in ["pubget", "ace"]:
        source_obj = getattr(study, source, None)
        if source_obj and source_obj.text:
            with open(source_obj.text) as f:
                text = f.read()
            break

    assert text is not None, "No text found for test study"

    extractor = UMLSDiseaseExtractor()

    # Test empty groups list
    empty_groups_input = {
        study_id: {
            "text": text,
            "participant_demographics": {"groups": []},
        }
    }
    results = extractor._transform(empty_groups_input)
    assert study_id in results
    assert results[study_id]["groups"] == []

    # Test missing diagnosis field
    no_diagnosis_demo = {"groups": [{"count": 10}]}
    no_diagnosis_input = {
        study_id: {
            "text": text,
            "participant_demographics": no_diagnosis_demo,
        }
    }
    results = extractor._transform(no_diagnosis_input)
    assert study_id in results
    assert results[study_id]["groups"] == []

    # Test invalid group data structure
    invalid_group_demo = {"groups": None}
    invalid_input = {
        study_id: {
            "text": text,
            "participant_demographics": invalid_group_demo,
        }
    }
    results = extractor._transform(invalid_input)
    assert study_id in results
    assert results[study_id]["groups"] == []

    # Test missing demographics entirely
    missing_demo_input = {
        study_id: {
            "text": text,
            "participant_demographics": None,
        }
    }
    results = extractor._transform(missing_demo_input)
    assert study_id in results
    assert results[study_id]["groups"] == []


@pytest.mark.skip(reason="UMLS tests are optional")
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
    extractor = UMLSDiseaseExtractor()

    # Set up pipeline kwargs
    pipeline_kwargs = {
        "participant_demographics": {
            "version": "1.0.0",
            "config_hash": "abc123",
            "pipeline_dir": tmp_path / "participant_demographics",
        }
    }

    # Run extraction - should raise error for missing results
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match=".*Missing results.json.*"):
        extractor.transform_dataset(
            sample_data, output_dir, input_pipeline_info=pipeline_kwargs
        )
