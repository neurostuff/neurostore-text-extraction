from pydantic import BaseModel, Field
from typing import List
from ns_extract.pipelines.api import APIEmbeddingExtractor
from ns_extract.pipelines.base import IndependentPipeline


class EmbeddingSchema(BaseModel):
    """Schema for TFIDF output"""

    embedding: List[float] = Field(description="Vector representation of the text")


class GeneralAPIEmbeddingExtractor(APIEmbeddingExtractor, IndependentPipeline):
    """Embedding extraction pipeline.

    Calculates embedding vectors for each document.
    """

    _version = "1.0.0"
    _output_schema = EmbeddingSchema
    _data_pond_inputs = {("pubget", "ace", "db"): ("text", "metadata")}
    _pipeline_inputs = {}
