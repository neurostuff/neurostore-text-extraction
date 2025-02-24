"""Extract task information from scientific papers."""
from .schemas import StudyMetadataModel
from .prompts import base_message
from ns_pipelines.pipeline import BasePromptPipeline


class TaskExtractor(BasePromptPipeline):
    """Task information extraction pipeline using LLM prompts."""

    _version = "1.0.0"
    _prompt = base_message  # Prompt template for extraction
    _output_schema = StudyMetadataModel  # Schema for validating final outputs
