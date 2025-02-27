"""Pipeline implementations for extracting information from papers."""
from .participant_demographics import ParticipantDemographicsExtractor
from .word_count import WordCountExtractor, WordDevianceExtractor
from .nv_task import TaskExtractor
from .base import Pipeline, IndependentPipeline, DependentPipeline
from .api import APIPromptExtractor

__all__ = [
    "ParticipantDemographicsExtractor",
    "WordCountExtractor",
    "WordDevianceExtractor",
    "TaskExtractor",
    "Pipeline",
    "IndependentPipeline",
    "DependentPipeline",
    "APIPromptExtractor",
]