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
        super().__init__()

    def _transform(self, inputs: dict, **kwargs) -> dict:
        """Run the word count extraction pipeline.

        Args:
            processed_inputs: Dictionary containing:
                - study_id: Unique identifier for the study
                    - text: Full text content (already loaded)
            **kwargs: Additional arguments including study_id

        Returns:
            Dictionary containing word count
        """
        word_counts = {
            study_id: {"word_count": len(inputs["text"].split())}
            for study_id, inputs in inputs.items()
        }  # Already loaded by InputManager
        return word_counts


class WordDevianceExtractor(Extractor, DependentPipeline):
    """Word deviance pipeline.

    Count the deviance of each study from the average word count.
    """

    _version = "1.0.0"
    _output_schema = WordDevianceSchema
    _data_pond_inputs = {("pubget", "ace"): ("text",)}
    _pipeline_inputs = {}

    def __init__(self, square_root=False):
        """Add any pipeline configuration here (as opposed to runtime arguments)"""
        self.square_root = square_root
        super().__init__()

    def _serialize_dataset_keys(self, dataset):
        """Create a string representation of dataset keys for hashing.

        Args:
            dataset: Dataset object containing study data

        Returns:
            String of sorted study IDs joined with underscores
        """
        return "_".join(sorted(dataset.data.keys()))

    def _transform(self, inputs: dict, **kwargs) -> dict:
        """Run the word deviance extraction pipeline.

        Args:
            processed_inputs: Dictionary mapping study IDs to their loaded inputs, where each has:
                - text: Text content string (already loaded)
            **kwargs: Additional arguments including study_id

        Returns:
            Dictionary mapping study IDs to their word count deviances
        """
        # Calculate word counts for all studies
        study_word_counts = {
            study_id: len(inputs["text"].split()) for study_id, inputs in inputs.items()
        }

        # Calculate average
        total_word_count = sum(study_word_counts.values())
        average_word_count = total_word_count // len(study_word_counts)

        # Calculate and return deviances
        return {
            study_id: {"word_deviance": abs(word_count - average_word_count)}
            for study_id, word_count in study_word_counts.items()
        }
