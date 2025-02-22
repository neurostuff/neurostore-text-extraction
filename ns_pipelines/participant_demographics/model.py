""" Extract participant demographics from articles. """
from . import prompts
from .clean import clean_prediction

from ns_pipelines.pipeline import BasePromptPipeline


class ParticipantDemographicsExtractor(IndependentPipeline):
    """Participant demographics extraction pipeline."""

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

        prompt_config = getattr(prompts, prompt_set)

        super().__init__(
            inputs=inputs, 
            input_sources=input_sources,
            env_variable=env_variable,
            env_file=env_file,
            prompt_set=prompt_set,
            extraction_model=extraction_model,
            prompt_config=prompt_config,
            kwargs=kwargs
        )


    def post_process(self, predictions):
        return clean_prediction(predictions)