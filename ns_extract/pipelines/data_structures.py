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
    """Information about the input pipeline."""

    pipeline_dir: Path = Field(description="Path to the pipeline directory")
    version: str = Field(description="Version of the pipeline")
    config_hash: str = Field(description="Hash of the pipeline configuration")


class PipelineOutputInfo(BaseModel):
    """Information about the pipeline output."""

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
    """Information about a study's processing results."""

    date: str = Field(description="When the study was processed")
    inputs: Dict[str, str] = Field(description="Input file paths and their MD5 hashes")
    valid: bool = Field(description="Whether outputs passed validation")
