"""Pipeline implementations for extracting information from papers."""

from .participant_demographics import ParticipantDemographicsExtractor
from .umls_disease import UMLSDiseaseExtractor
from .nv_task import TaskExtractor
from .base import Pipeline, IndependentPipeline, DependentPipeline
from .api import APIPromptExtractor
from .tfidf import TFIDFExtractor

__all__ = [
    "ParticipantDemographicsExtractor",
    "TaskExtractor",
    "Pipeline",
    "IndependentPipeline",
    "DependentPipeline",
    "APIPromptExtractor",
    "TFIDFExtractor",
    "UMLSDiseaseExtractor",
]
