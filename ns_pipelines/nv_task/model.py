"""Extract task information from scientific papers."""
from .schemas import StudyMetadataModel
from ns_pipelines.pipeline import BasePromptPipeline

class TaskExtractor(BasePromptPipeline):
    """Task information extraction pipeline using LLM prompts."""

    _version = "1.0.0"
    _schema = StudyMetadataModel  # Pydantic schema for validation
    _prompt = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

Your task is to identify information about the design of the fMRI task and analysis of the neuroimaging data.
If any information is missing or not explicitly stated in the text, return `null` for that field.

For any extracted text, maintain fidelity to the source. Avoid inferring information not explicitly stated. If a field cannot be completed, return `null`.

Text sample: ${text}
"""

    def pre_process(self, text: str) -> str:
        """Pre-process the text before extraction if needed.
        
        Args:
            text: Raw text to process
            
        Returns:
            Processed text
        """
        # Currently no pre-processing needed, but method available for future use
        return text
