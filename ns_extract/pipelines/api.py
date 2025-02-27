from typing import Any, Dict, Optional, Type
import logging
import os
from pydantic import BaseModel
from openai import OpenAI
from .base import IndependentPipeline
from publang.extract import extract_from_text


class APIPromptExtractor(IndependentPipeline):
    """Pipeline that uses a prompt and a pydantic schema to extract information from text."""
    
    # Class attributes to be defined by subclasses
    _prompt: str = None  # Prompt template for extraction
    _extraction_schema: Type[BaseModel] = None  # Schema used for LLM extraction (if not defined, uses _output_schema)

    def __init__(
        self,
        extraction_model: str,
        inputs: tuple = ("text",),
        input_sources: tuple = ("pubget", "ace"),
        env_variable: Optional[str] = None,
        env_file: Optional[str] = None,
        client_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the prompt-based pipeline.
        
        Args:
            extraction_model: Model to use for extraction (e.g., 'gpt-4')
            inputs: Input types required
            input_sources: Valid input sources
            env_variable: Environment variable containing API key
            env_file: Path to file containing API key
            client_url: Optional URL for OpenAI client
            **kwargs: Additional arguments for the completion function
        """
        if not self._prompt:
            raise ValueError("Subclass must define _prompt template")
        if not self._extraction_schema:
            self._extraction_schema = self._output_schema

        super().__init__(inputs=inputs, input_sources=input_sources)
        self.extraction_model = extraction_model
        self.env_variable = env_variable
        self.env_file = env_file
        self.client_url = client_url
        self.kwargs = kwargs

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

    def execute(self, inputs: Dict[str, Any], **kwargs) -> dict:
        """Execute LLM-based extraction using processed inputs.
        
        Args:
            processed_inputs: Dictionary containing processed text and initialized client
            **kwargs: Additional arguments (like n_cpus)
            
        Returns:
            Raw predictions from LLM
        """
        # Initialize client
        client = self._load_client()

        # Read 
        with open(inputs['text'], 'r') as f:
            text = f.read()

        # Create chat completion configuration
        completion_config = {
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt.replace("${text}", text) + "\n Call the extractData function to save the output."
                }
            ],
            "output_schema": self._extraction_schema.model_json_schema()
        }
        if self.kwargs:
            completion_config.update(self.kwargs)

        # Extract predictions
        results = extract_from_text(
            text,
            model=self.extraction_model,
            client=client,
            **completion_config
        )

        if not results:
            logging.warning("No results found")
            return None
            
        return results
