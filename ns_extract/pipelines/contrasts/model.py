"""Extract brain coordinates from contrasts in scientific papers."""

from .schemas import ContrastBase
from .prompts import base_message
from ns_extract.pipelines.api import APIPromptExtractor


class CoordinatesExtractor(APIPromptExtractor):
    """Task information extraction pipeline using LLM prompts."""

    _version = "1.1.0"
    _prompt = base_message
    _output_schema = ContrastBase