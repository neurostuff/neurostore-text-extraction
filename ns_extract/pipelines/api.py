from typing import Optional, Type, Literal
import logging
import os
import numpy as np
from pydantic import BaseModel
from openai import OpenAI
from .base import IndependentPipeline, Extractor
from publang.extract import extract_from_text
import tiktoken


class APIBaseExtractor(Extractor, IndependentPipeline):
    """
    Base class for API-based extractors. Handles API key retrieval and client initialization.
    """

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


class APIPromptExtractor(APIBaseExtractor):
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
        if not self._prompt:
            raise ValueError("Subclass must define _prompt template")
        if not self._extraction_schema:
            self._extraction_schema = self._output_schema
        super().__init__(
            extraction_model=extraction_model,
            env_variable=env_variable,
            env_file=env_file,
            client_url=client_url,
            disable_abbreviation_expansion=disable_abbreviation_expansion,
            **kwargs,
        )

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


TEXT_MAPPING = {
    "full_text": "text",
    "abstract": "metadata.abstract",
    "title": "metadata.title",
}

MAX_TOKENS = 8192

MINIMUM_CHUNK_SIZE = 5


def get_nested_value(data: dict, key_path: str):
    """Access nested dictionary values using dot notation."""
    keys = key_path.split(".")
    for key in keys:
        data = data[key]
    return data


class APIEmbeddingExtractor(APIBaseExtractor):
    """
    Pipeline that uses an embedding model to extract vector representations from text.
    """

    _extraction_schema: Type[BaseModel] = None  # Schema used for LLM extraction
    _data_pond_inputs = {("pubget", "ace"): ("text", "metadata")}
    _pipeline_inputs = {}

    def __init__(
        self,
        extraction_model: str,
        text_source: Literal["full_text", "abstract", "title"] = "full_text",
        env_variable: Optional[str] = None,
        env_file: Optional[str] = None,
        client_url: Optional[str] = None,
        disable_abbreviation_expansion: bool = False,
        **kwargs,
    ):
        self.text_source = text_source
        super().__init__(
            extraction_model=extraction_model,
            env_variable=env_variable,
            env_file=env_file,
            client_url=client_url,
            disable_abbreviation_expansion=disable_abbreviation_expansion,
            **kwargs,
        )

    def _transform(self, inputs: dict, **kwargs) -> dict:
        """
        Extract embeddings from text using the specified model.
        Args:
            inputs: Dictionary containing text for each study_id.
        Returns:
            Dictionary mapping study_id to averaged embeddings.
        """

        results = {}
        text_source = TEXT_MAPPING[self.text_source]
        enc = tiktoken.encoding_for_model(self.extraction_model)

        def chunk_paragraph(paragraph, max_tokens=MAX_TOKENS):
            tokens = enc.encode(paragraph)
            if len(tokens) <= max_tokens and len(tokens) >= MINIMUM_CHUNK_SIZE:
                return [paragraph]
            if len(tokens) < MINIMUM_CHUNK_SIZE:
                return []
            # Use spaCy to split into sentences
            if not self._nlp_initialized:
                self._init_nlp_components()
            doc = self._nlp(paragraph)
            sentences = [sent.text for sent in doc.sents]
            chunks = []
            current_chunk = ""
            for sent in sentences:
                test_chunk = current_chunk + " " + sent if current_chunk else sent
                if len(enc.encode(test_chunk)) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Filter out chunks that are too short
            chunks = [chunk for chunk in chunks if len(chunk) >= MINIMUM_CHUNK_SIZE]
            return chunks

        for study_id, study_inputs in inputs.items():
            text = get_nested_value(study_inputs, text_source)
            paragraphs = text.split("\n\n")
            all_chunks = []
            for para in paragraphs:
                all_chunks.extend(chunk_paragraph(para, MAX_TOKENS))
            embeddings = []
            for chunk in all_chunks:
                embedding_response = self.client.embeddings.create(
                    input=chunk,
                    model=self.extraction_model,
                    **self.kwargs,
                )
                embedding = embedding_response.data[0].embedding
                embeddings.append(np.array(embedding))
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0).tolist()
            else:
                avg_embedding = []
            results[study_id] = {"embedding": avg_embedding}
        return results
