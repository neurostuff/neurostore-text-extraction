"""Patient Study Extractor that checks if any group is labeled as 'patient'."""

from typing import Dict, Any

from pydantic import BaseModel, Field

from ns_extract.pipelines.base import (
    Extractor,
    IndependentPipeline,
)


class PatientStudyOutput(BaseModel):
    """Schema for patient study output."""

    patient_study: bool = Field(
        ..., description="True if any group is labeled as 'patient', False otherwise"
    )


class PatientStudyExtractor(Extractor, IndependentPipeline):
    """Extractor to identify if a study has a patient group."""

    _version = "1.0.0"
    _output_schema = PatientStudyOutput
    _input_pipelines = {("participant_demographics",): ("results",)}

    def __init__(self):
        """Initialize extractor."""
        super().__init__()

    def _transform(self, inputs: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Transform input data into output format.

        Args:
            inputs: Dict mapping study IDs to their input data
                   Each study's data includes:
                   - participant_demographics.results: Demographics results data
            **kwargs: Additional transformation arguments

        Returns:
            Dict mapping study IDs to their patient_study results
        """
        demographics = inputs.get("participant_demographics", {})

        # Check each group name for "patient"
        has_patient_group = any(
            group.get("name", "").lower() == "patient"
            for group in demographics.get("groups", [])
        )

        return {"patient_study": has_patient_group}
