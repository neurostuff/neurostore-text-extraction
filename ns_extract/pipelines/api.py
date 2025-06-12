from typing import Optional, Type
import logging
import os
from pydantic import BaseModel
from openai import OpenAI
from .base import IndependentPipeline, Extractor
from publang.extract import extract_from_text


class APIPromptExtractor(Extractor, IndependentPipeline):
    """Pipeline that uses a prompt and a pydantic schema to extract information from text."""

    _prompt: str = None  # Prompt template for extraction
    _extraction_schema: Type[BaseModel] = None  # Schema used for LLM extraction
    _data_pond_inputs = {("pubget", "ace"): ("text",)}
    _pipeline_inputs = {}

    def __init__(
        self,
        extraction_model: str,
        env_variable: Optional[str] = None,
        env_file: Optional[str] = None,
        client_url: Optional[str] = None,
        disable_abbreviation_expansion: bool = False,
        **kwargs,
    ):
        """Initialize the prompt-based pipeline.

        Args:
            extraction_model: Model to use for extraction (e.g., 'gpt-4')
            env_variable: Environment variable containing API key
            env_file: Path to file containing API key
            client_url: Optional URL for OpenAI client
            disable_abbreviation_expansion: If True, disables abbreviation expansion
            **kwargs: Additional arguments for the OpenAI completion function
        """
        if not self._prompt:
            raise ValueError("Subclass must define _prompt template")
        if not self._extraction_schema:
            self._extraction_schema = self._output_schema

        self.extraction_model = extraction_model
        self.env_variable = env_variable
        self.env_file = env_file
        self.client_url = client_url
        self.kwargs = kwargs

        # Initialize OpenAI client
        self.client = self._load_client()

        super().__init__(disable_abbreviation_expansion=disable_abbreviation_expansion)

    def _load_client(self) -> OpenAI:
        """Load the OpenAI client.

        Returns:
            OpenAI client instance

        Raises:
            ValueError: If no API key provided or unsupported model
        """
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("No API key provided")

        return OpenAI(api_key=api_key, base_url=self.client_url)

    def _get_api_key(self) -> Optional[str]:
        """Read the API key from environment variable or file.

        Returns:
            API key if found, None otherwise
        """
        if self.env_variable:
            api_key = os.getenv(self.env_variable)
            if api_key:
                return api_key

        if self.env_file:
            try:
                with open(self.env_file) as f:
                    key_parts = f.read().strip().split("=")
                    if len(key_parts) == 2:
                        return key_parts[1]
                    logging.warning("Invalid format in API key file")
            except FileNotFoundError:
                logging.error(f"API key file not found: {self.env_file}")

        return None

    def _transform(self, inputs: dict, **kwargs) -> dict:
        """Execute LLM-based extraction using processed inputs.

        Args:
            processed_inputs: Dictionary containing:
                   - text: Full text content (already loaded)
            **kwargs: Additional arguments (like study_id)

        Returns:
            Raw predictions from LLM
        """
        results = {}
        for study_id, study_inputs in inputs.items():
            # Get text content - already loaded by InputManager
            text = study_inputs["text"]

            # Create chat completion configuration
            completion_config = {
                "messages": [
                    {
                        "role": "user",
                        "content": self._prompt
                        + "\n Call the extractData function to save the output.",
                    }
                ],
                "output_schema": self._extraction_schema.model_json_schema(),
            }
            if self.kwargs:
                completion_config.update(self.kwargs)

            # Replace $ with $$ to escape $ signs in the prompt
            # (otherwise interpreted as a special character by Template())
            text = text.replace("$", "$$")

            # Extract predictions
            study_results = extract_from_text(
                text,
                model=self.extraction_model,
                client=self.client,
                **completion_config,
            )

            if not study_results:
                logging.warning(
                    f"No results found for study {study_id} with model {self.extraction_model}"
                )

            results[study_id] = study_results

        return results
