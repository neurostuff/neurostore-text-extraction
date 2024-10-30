from datetime import datetime
import hashlib
import json

from ns_pipelines.pipeline import IndependentPipeline, DependentPipeline


class WordCountExtraction(IndependentPipeline):
    """Word count extraction pipeline."""

    _version = "1.0.0"
    _hash_args = ["_inputs", "_input_sources"]
    _pipeline_type = "independent"

    def function(self, study_inputs):
        """Run the word count extraction pipeline."""
        text_file = study_inputs["text"]

        with open(text_file, "r") as f:
            text = f.read()

        return {"word_count": len(text.split())}


class WordDevianceExtraction(DependentPipeline):
    """Word deviance pipeline.

    Count the deviance of each study from the average word count.
    """

    _version = "1.0.0"
    _hash_args = ["_inputs", "_input_sources"]
    _pipeline_type = "dependent"

    def group_function(self, all_study_inputs):
        """Run the word count extraction pipeline."""

        # Calculate the average word count
        total_word_count = 0
        total_studies = len(all_study_inputs)
        study_word_counts = {}
        for study_id, study_inputs in all_study_inputs.items():
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

        # key is study_id, value is deviance from average word count
        return study_word_deviances
