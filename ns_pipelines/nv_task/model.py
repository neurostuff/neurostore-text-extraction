""" Extract participant demographics from HTML files. """
import os

from publang.extract import extract_from_text
from openai import OpenAI
import logging

from . import prompts

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

    return predictions


def _load_client(model_name, api_key):
    if 'gpt' in model_name:
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Model {model_name} not supported")

    return client


def _load_prompt_config(prompt_set):
    return getattr(prompts, prompt_set)


class TaskExtractor(IndependentPipeline):
    """Task information extraction pipeline."""

    _version = "1.0.0"

    def __init__(
        self,
        extraction_model,
        prompt_set,
        inputs=("text",),
        input_sources=("pubget", "ace"),
        env_variable=None,
        env_file=None,
        **kwargs
    ):
        super().__init__(inputs=inputs, input_sources=input_sources)
        self.extraction_model = extraction_model
        self.prompt_set = prompt_set
        self.env_variable = env_variable
        self.env_file = env_file
        self.kwargs = kwargs

    def get_api_key(self):
        """Read the API key from the environment variable or file."""
        if self.env_variable:
            api_key = os.getenv(self.env_variable)
            if api_key is not None:
                return api_key
        if self.env_file:
            with open(self.env_file) as f:
                return ''.join(f.read().strip().split("=")[1])
        else:
            raise ValueError("No API key provided")

    def _run(self, study_inputs, n_cpus=1):
        """Run the participant demographics extraction pipeline."""
        api_key = self.get_api_key()
        extraction_client = _load_client(self.extraction_model, api_key)

        prompt_config = _load_prompt_config(self.prompt_set)
        if self.kwargs is not None:
            prompt_config.update(self.kwargs)

        with open(study_inputs["text"]) as f:
            text = f.read()

        predictions = extract(
            self.extraction_model,
            extraction_client,
            text,
            prompt_set=self.prompt_set,
            **prompt_config
        )

        # Save predictions
        return {"predictions": predictions}
