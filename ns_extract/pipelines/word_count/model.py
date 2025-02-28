from pydantic import BaseModel, Field
from ns_extract.pipelines.base import IndependentPipeline, DependentPipeline


class WordCountSchema(BaseModel):
    word_count: int = Field(description="Number of words in the text")


class WordDevianceSchema(BaseModel):
    word_deviance: int = Field(description="Absolute difference from average word count")


class WordCountExtractor(IndependentPipeline):
    """Word count extraction pipeline."""

    _version = "1.0.0"
    _output_schema = WordCountSchema

    def __init__(self, inputs=("text",), input_sources=("pubget", "ace"), square_root=False):
        """Add any pipeline configuration here (as opposed to runtime arguments)"""
        self.square_root = square_root
        super().__init__(inputs=inputs, input_sources=input_sources)

    def execute(self, processed_inputs: dict, **kwargs) -> dict:
        """Run the word count extraction pipeline.
        
        Args:
            processed_inputs: Dictionary containing processed text
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing word count
        """
        text_file = processed_inputs["text"]

        with open(text_file, "r") as f:
            text = f.read()

        return {"word_count": len(text.split())}


class WordDevianceExtractor(DependentPipeline):
    """Word deviance pipeline.

    Count the deviance of each study from the average word count.
    """

    _version = "1.0.0"
    _output_schema = WordDevianceSchema

    def __init__(self, inputs=("text",), input_sources=("pubget", "ace"), square_root=False):
        self.square_root = square_root
        super().__init__(inputs=inputs, input_sources=input_sources)

    def execute(self, processed_inputs: dict, **kwargs) -> dict:
        """Run the word deviance extraction pipeline.
        
        Args:
            processed_inputs: Dictionary containing all study inputs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping study IDs to their word count deviances
        """
        # Calculate the average word count
        total_word_count = 0
        total_studies = len(processed_inputs)
        study_word_counts = {}

        for study_id, study_inputs in processed_inputs.items():
            text_file = study_inputs["text"]

            with open(text_file, "r") as f:
                text = f.read()

            num_words = len(text.split())
            total_word_count += num_words
            study_word_counts[study_id] = num_words

        average_word_count = total_word_count // total_studies
        study_word_deviances = {
            study_id: {"word_deviance": abs(num_words - average_word_count)}
            for study_id, num_words in study_word_counts.items()
        }

        return study_word_deviances
