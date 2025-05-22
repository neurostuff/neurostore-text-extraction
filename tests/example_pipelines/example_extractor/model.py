"""Example Extractor module demonstrating pipeline and datapond input handling."""

from typing import Dict, Any

from pydantic import BaseModel, Field

from ns_extract.pipelines.base import (
    Extractor,
    DependentPipeline,
)


class ExampleOutput(BaseModel):
    """Schema for example extractor output."""

    value: str = Field(
        ...,
        description="Example extracted value from input data",
        json_schema_extra={
            "normalize_text": True,
            "expand_abbreviations": True
        }
    )
    confidence: float = Field(
        ...,
        description="Confidence score of the extraction",
        ge=0.0,
        le=1.0
    )


class ExampleExtractor(Extractor, DependentPipeline):
    """Extractor demonstrating pipeline and datapond input handling.

    This extractor shows how to:
    - Process pipeline inputs (from participant_demographics)
    - Handle datapond data
    - Produce structured output
    """

    _version = "1.0.0"
    _output_schema = ExampleOutput
    _input_pipelines = {("participant_demographics",): ("result",)}
    _data_pond_inputs = {
        ("pubget", "ace"): ("text",),
    }

    def __init__(self):
        """Initialize extractor."""
        super().__init__()

    def _transform(
        self, inputs: Dict[str, Dict[str, Any]], **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Transform input data into output format.

        Args:
            inputs: Dict mapping study IDs to their input data.
                   Each study's data includes:
                   - participant_demographics.results: Demographics results data
                   - text: Text data from pubget/ace sources
            **kwargs: Additional transformation arguments

        Returns:
            Dict mapping study IDs to their extraction results containing:
            - value: Extracted value combining demographics and text data
            - confidence: Confidence score of extraction
        """
        results = {}

        # Process each study
        for study_id, study_data in inputs.items():
            # Get pipeline inputs
            demographics = study_data.get("participant_demographics", {})
            groups = demographics.get("groups", [])

            # Get text data from data pond
            text_data = study_data.get("text", "")

            # Example processing logic
            if groups and text_data:
                # Combine demographics and text data
                value = f"{groups[0].get('name', '')}-{text_data[:150]}"  # Take first 50 chars
                confidence = 1.0
            elif groups:
                # Use only demographics data
                value = groups[0].get("name", "")
                confidence = 0.7
            elif text_data:
                # Use only text data
                value = text_data[:150]  # Take first 50 chars
                confidence = 0.5
            else:
                value = "no_data"
                confidence = 0.0

            results[study_id] = {"value": value, "confidence": confidence}

        return results
