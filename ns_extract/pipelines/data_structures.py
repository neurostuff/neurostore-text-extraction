"""Data structure definitions for pipeline metadata and configuration.

This module defines Pydantic models for:
- Pipeline input/output metadata
- Study processing results
- Configuration schemas
"""

from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel, Field


class InputPipelineInfo(BaseModel):
    """Information about an input pipeline dependency.

    Attributes:
        pipeline_dir: Path to the pipeline's output directory
        version: Version identifier of the pipeline
        config_hash: Hash of the pipeline's configuration
    """

    pipeline_dir: Path = Field(description="Path to the pipeline directory")
    version: str = Field(description="Version of the pipeline")
    config_hash: str = Field(description="Hash of the pipeline configuration")


class PipelineOutputInfo(BaseModel):
    """Metadata about a pipeline's output.

    This includes configuration details, versions, and input dependencies
    that are needed to reproduce the pipeline's results.

    Attributes:
        date: ISO format date when output was generated
        version: Pipeline version identifier
        config_hash: Hash of pipeline configuration
        extractor: Name of extractor class used
        extractor_kwargs: Arguments passed to extractor
        transform_kwargs: Arguments passed to transform
        input_pipelines: Dependencies on other pipeline outputs
        schema: JSON schema of the output data format
    """

    date: str = Field(description="Date of the output")
    version: str = Field(description="Version of the pipeline")
    config_hash: str = Field(description="Hash of the pipeline configuration")
    extractor: str = Field(description="Name of the extractor used")
    extractor_kwargs: Dict[str, Any] = Field(
        description="Arguments passed to the extractor"
    )
    transform_kwargs: Dict[str, Any] = Field(
        description="Arguments passed to the transform function"
    )
    input_pipelines: Dict[str, InputPipelineInfo] = Field(
        description="Pipelines used as inputs to this pipeline"
    )
    schema: Dict[str, Any] = Field(description="Schema of the output data")


class StudyOutputJson(BaseModel):
    """Information about the processing results for a single study.

    Attributes:
        date: ISO format timestamp of processing
        inputs: Mapping of input files to their MD5 hashes
        valid: Whether the outputs passed validation
    """

    date: str = Field(description="When the study was processed")
    inputs: Dict[str, str] = Field(description="Input file paths and their MD5 hashes")
    valid: bool = Field(description="Whether outputs passed validation")
