""" Extract participant demographics from HTML files. """
import os

from publang.extract import extract_from_text
from openai import OpenAI
import logging

from . import prompts
from .clean import clean_prediction

from ns_pipelines.pipeline import IndependentPipeline


def extract(extraction_model, extraction_client, text, prompt_set='', **extract_kwargs):
    extract_kwargs.pop('search_query', None)

    # Extract
    predictions = extract_from_text(
        text,
        model=extraction_model,
        client=extraction_client,
        **extract_kwargs
    )

    if not predictions:
        logging.warning("No predictions found.")
        return None, None

    clean_preds = clean_prediction(predictions)

    return predictions, clean_preds


def _load_client(model_name):
    if 'gpt' in model_name:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    else:
        raise ValueError(f"Model {model_name} not supported")

    return client


def _load_prompt_config(prompt_set):
    return getattr(prompts, prompt_set)


class ParticipantDemographicsExtraction(IndependentPipeline):
    """Participant demographics extraction pipeline."""

    _version = "1.0.0"

    def __init__(
        self,
        extraction_model,
        prompt_set,
        inputs=("text",),
        input_sources=("pubget", "ace"),
        **kwargs
    ):
        super().__init__(inputs=inputs, input_sources=input_sources)
        self.extraction_model = extraction_model
        self.prompt_set = prompt_set
        self.kwargs = kwargs

    def _run(self, study_inputs, n_cpus=1):
        """Run the participant demographics extraction pipeline."""
        extraction_client = _load_client(self.extraction_model)

        prompt_config = _load_prompt_config(self.prompt_set)
        if self.kwargs is not None:
            prompt_config.update(self.kwargs)

        with open(study_inputs["text"]) as f:
            text = f.read()

        predictions, clean_preds = extract(
            self.extraction_model,
            extraction_client,
            text,
            prompt_set=self.prompt_set,
            **prompt_config
        )

        # Save predictions
        return {"predictions": predictions, "clean_predictions": clean_preds}
