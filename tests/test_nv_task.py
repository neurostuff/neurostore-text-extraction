import json
import pytest

from ns_extract.pipelines import TaskExtractor
from ns_extract.dataset import Dataset
from ns_extract.pipelines.nv_task.schemas import (
    StudyMetadataModel,
    fMRITaskMetadataModel,
    TaskMetadataModel,
)


@pytest.mark.vcr(record_mode="once", filter_headers=["authorization"])
def test_TaskExtractor(sample_data, tmp_path):
    """Test the task extraction pipeline."""
    # Initialize extractor
    task_extractor = TaskExtractor(
        extraction_model="gpt-4o-mini-2024-07-18",
        env_variable="OPENAI_API_KEY",
    )
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "task_extraction"

    # Initial run
    task_extractor.transform_dataset(dataset, output_dir)

    # Verify directory structure and files
    version_dir = next(output_dir.glob("TaskExtractor/1.0.0/*"))
    assert version_dir.exists()

    # Check pipeline info
    pipeline_info = json.loads((version_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == "1.0.0"
    assert pipeline_info["type"] == "apipromptextractor"
    assert pipeline_info["arguments"]["extraction_model"] == "gpt-4o-mini-2024-07-18"

    # Verify study outputs and schema validation
    for study_dir in version_dir.glob("*"):
        if study_dir.is_dir():
            results_file = study_dir / "results.json"
            info_file = study_dir / "info.json"
            assert results_file.exists()
            assert info_file.exists()

            # Load and validate results
            study_results = json.loads(results_file.read_text())

            # Validate against StudyMetadataModel
            validated = StudyMetadataModel.model_validate(study_results)

            # Check required fields
            assert isinstance(validated.Modality, list)
            assert all(
                mod
                in [
                    "fMRI-BOLD",
                    "StructuralMRI",
                    "DiffusionMRI",
                    "PET FDG",
                    "PET [15O]-water",
                    "fMRI-CBF",
                    "fMRI-CBV",
                    "MEG",
                    "EEG",
                    "Other",
                ]
                for mod in validated.Modality
            )

            # Check fMRI tasks
            if validated.fMRITasks:
                for task in validated.fMRITasks:
                    assert isinstance(task, fMRITaskMetadataModel)
                    assert task.TaskName
                    assert task.TaskDescription
                    assert task.DesignDetails
                    assert isinstance(task.RestingState, bool)
                    assert task.TaskDesign and all(
                        design in ["Blocked", "EventRelated", "Mixed", "Other"]
                        for design in task.TaskDesign
                    )

            # Check behavioral tasks
            if validated.BehavioralTasks:
                for task in validated.BehavioralTasks:
                    assert isinstance(task, TaskMetadataModel)
                    assert task.TaskName
                    assert task.TaskDescription
                    assert task.DesignDetails
