"""Extract task information from scientific papers."""
from .schemas import StudyMetadataModel
from .prompts import base_message
from ns_pipelines.pipeline import BasePromptPipeline



class TaskExtractor(BasePromptPipeline):
    """Task information extraction pipeline using LLM prompts."""

    _version = "1.0.0"
    _schema = StudyMetadataModel  # Pydantic schema for validation
    _prompt = base_message  # Prompt for the pipeline