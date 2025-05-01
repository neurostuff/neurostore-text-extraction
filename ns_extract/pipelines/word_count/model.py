from pydantic import BaseModel, Field
from ns_extract.pipelines.base import IndependentPipeline, DependentPipeline, Extractor


class WordCountSchema(BaseModel):
    word_count: int = Field(description="Number of words in the text")


class WordDevianceSchema(BaseModel):
    word_deviance: int = Field(
        description="Absolute difference from average word count"
    )


class WordCountExtractor(Extractor, IndependentPipeline):
    """Word count extraction pipeline."""

    _version = "1.0.0"
    _output_schema = WordCountSchema
    _data_pond_inputs = {("pubget", "ace"): ("text",)}
    _pipeline_inputs = {}

    def __init__(
        self,
        square_root=False,
    ):
        """Add any pipeline configuration here (as opposed to runtime arguments)"""
        self.square_root = square_root
        IndependentPipeline.__init__(self, extractor=self)

    def _transform(self, processed_inputs: dict, **kwargs) -> dict:
        """Run the word count extraction pipeline.

        Args:
            processed_inputs: Dictionary containing:
                - text: Full text content (already loaded)
            **kwargs: Additional arguments including study_id

        Returns:
            Dictionary containing word count
        """
        text = processed_inputs["text"]  # Already loaded by InputManager
        return {"word_count": len(text.split())}


class WordDevianceExtractor(DependentPipeline):
    """Word deviance pipeline.

    Count the deviance of each study from the average word count.
    """

    _version = "1.0.0"
    _output_schema = WordDevianceSchema

    def __init__(
        self, inputs=("text",), input_sources=("pubget", "ace"), square_root=False
    ):
        self.square_root = square_root
        super().__init__(inputs=inputs, input_sources=input_sources)

    def execute(self, processed_inputs: dict, **kwargs) -> dict:
        """Run the word deviance extraction pipeline.

        Args:
            processed_inputs: Dictionary containing all study inputs
                            Each study's data includes:
                            - text: Full text content (already loaded)
            **kwargs: Additional arguments including study_id

        Returns:
            Dictionary mapping study IDs to their word count deviances
        """
        # Calculate word counts for all studies
        study_word_counts = {
            study_id: len(study_inputs["text"].split())
            for study_id, study_inputs in processed_inputs.items()
        }

        # Calculate average
        total_word_count = sum(study_word_counts.values())
        average_word_count = total_word_count // len(study_word_counts)

        # Calculate deviances
        study_word_deviances = {
            study_id: {"word_deviance": abs(num_words - average_word_count)}
            for study_id, num_words in study_word_counts.items()
        }

        return study_word_deviances
