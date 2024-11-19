
from ns_pipelines.pipeline import IndependentPipeline, DependentPipeline


class WordCountExtraction(IndependentPipeline):
    """Word count extraction pipeline."""

    _version = "1.0.0"

    def __init__(self, inputs=("text",), input_sources=("pubget", "ace"), square_root=False):
        """add any pipeline configuration here (as opposed to runtime arguments like n_cpus or n_cores)"""
        self.square_root = square_root
        super().__init__(inputs=inputs, input_sources=input_sources)

    def _run(self, study_inputs, debug=False):
        """Run the word count extraction pipeline."""
        text_file = study_inputs["text"]

        if debug:
            print(f"Processing {text_file}")

        with open(text_file, "r") as f:
            text = f.read()

        return {"word_count": len(text.split())}


class WordDevianceExtraction(DependentPipeline):
    """Word deviance pipeline.

    Count the deviance of each study from the average word count.
    """

    _version = "1.0.0"

    def __init__(self, inputs=("text",), input_sources=("pubget", "ace"), square_root=False):
        self.square_root = square_root
        super().__init__(inputs=inputs, input_sources=input_sources)

    def _run(self, all_study_inputs, debug=False):
        """Run the word deviance extraction pipeline."""

        # Calculate the average word count
        total_word_count = 0
        total_studies = len(all_study_inputs)
        study_word_counts = {}

        if debug:
            print(f"Processing {total_studies} studies")

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
