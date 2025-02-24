from datetime import datetime
import inspect
import json
import hashlib
import os
import logging
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Type

from pydantic import BaseModel
from openai import OpenAI

from publang.extract import extract_from_text


INPUTS = [
    "text",
    "coordinates",
    "metadata",
]

RAW_INPUTS = [
    "raw.html",
    "raw.xml",
    "raw.tables",
    "raw.tables_xml",
]


def deep_getattr(obj: Any, attr_path: str, default: Any = None) -> Any:
    try:
        return reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return default


class FileManager:
    """Utility class for file handling operations."""

    @staticmethod
    def calculate_md5(file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        with file_path.open('r') as f:
            file_contents = f.read()
        return hashlib.md5(file_contents.encode()).hexdigest()

    @staticmethod
    def load_json(file_path: Path) -> Dict:
        """Load JSON from a file."""
        with file_path.open('r') as f:
            return json.load(f)

    @staticmethod
    def write_json(file_path: Path, data: Dict):
        """Write JSON to a file."""
        with file_path.open('w') as f:
            json.dump(data, f)

    @staticmethod
    def get_next_available_dir(base_path: Path) -> Path:
        """Find the next available directory by appending numbers (-1, -2, etc.) if necessary."""
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(f"{base_path.name}-{counter}")
            counter += 1
        return new_path


class Pipeline(ABC):
    """Abstract pipeline class for processing data."""

    _version: str = None
    _schema: Type[BaseModel] = None  # Required schema for validation

    def __init__(self, inputs: Union[tuple, list] = ("text",), input_sources: tuple = ("pubget", "ace")):
        if not self._schema:
            raise ValueError("Subclass must define _schema class")
            
        self.inputs = inputs
        self.input_sources = input_sources
        self._pipeline_type = inspect.getmro(self.__class__)[1].__name__.lower().rstrip("pipeline")

    @abstractmethod
    def process_dataset(self, dataset: Any, output_directory: Path, **kwargs):
        """Process a full dataset through the pipeline."""
        pass

    def _process_inputs(self, study_inputs: Dict[str, Any], **kwargs) -> Dict:
        """Process inputs and validate outputs."""
        try:
            results = self.run(study_inputs, **kwargs)
            if results:
                results = self.validate_predictions(results)
            return results
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            return None
            
    @abstractmethod
    def run(self, study_inputs: Dict[str, Any], **kwargs) -> Dict:
        """Run the core pipeline logic. To be implemented by subclasses."""
        pass

    def validate_predictions(self, predictions: dict) -> Optional[dict]:
        """Validate predictions against the schema.
        
        Args:
            predictions: Raw predictions from pipeline
            
        Returns:
            Validated predictions or None if validation fails
        """
        try:
            validated = self._schema.model_validate(predictions)
            return validated.model_dump()
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return None

    def create_directory_hash(self, dataset: Any) -> str:
        """Create a hash for the dataset."""
        dataset_str = self._serialize_dataset_keys(dataset)
        arg_str = self._serialize_pipeline_args()
        return hashlib.shake_256(f"{dataset_str}_{arg_str}".encode()).hexdigest(6)

    def filter_inputs(self, output_directory: Path, dataset: Any) -> bool:
        """Filter inputs based on the pipeline type."""
        existing_results = self._filter_existing_results(output_directory, dataset)
        matching_results = self._identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        keep_ids = set(dataset.data.keys()) - {db_id for db_id, match in matching_results.items() if match}
        return dataset.slice(keep_ids)

    def gather_all_study_inputs(self, dataset: Any) -> Dict[str, Dict[str, Path]]:
        """Collect all inputs for the dataset."""
        return {db_id: self.collect_study_inputs(study) for db_id, study in dataset.data.items()}

    def collect_study_inputs(self, study: Any) -> Dict[str, Path]:
        """Collect inputs for a study."""
        study_inputs = {}
        for source in self.input_sources:
            source_obj = getattr(study, source, None)
            if source_obj:
                for input_type in self.inputs:
                    input_obj = deep_getattr(source_obj, input_type, None)
                    if input_obj and study_inputs.get(input_type) is None:
                        study_inputs[input_type] = input_obj
        return study_inputs

    def write_pipeline_info(self, hash_outdir: Path):
        """Write information about the pipeline to a pipeline_info.json file."""
        pipeline_info = {
            "date": datetime.now().isoformat(),
            "version": self._version,
            "type": self._pipeline_type,
            "arguments": {
                arg: getattr(self, arg) for arg in inspect.signature(self.__init__).parameters.keys()
            },
        }
        FileManager.write_json(hash_outdir / "pipeline_info.json", pipeline_info)

    def write_study_info(self, hash_outdir: Path, db_id: str, study_inputs: Dict[str, Path]):
        """Write information about the current run to an info.json file."""
        output_info = {
            "date": datetime.now().isoformat(),
            "inputs": {str(input_file): FileManager.calculate_md5(input_file) for input_file in study_inputs.values()}
        }
        FileManager.write_json(hash_outdir / db_id / "info.json", output_info)

    def _serialize_dataset_keys(self, dataset: Any) -> str:
        """Return a hashable string of the input dataset."""
        return "_".join(list(dataset.data.keys()))

    def _serialize_pipeline_args(self) -> str:
        """Return a hashable string of the arguments."""
        args = list(inspect.signature(self.__init__).parameters.keys())
        return '_'.join([f"{arg}_{str(getattr(self, arg))}" for arg in args])

    def _filter_existing_results(self, output_dir: Path, dataset: Any) -> Dict[str, Dict]:
        """Find the most recent result for an existing study."""
        existing_results = {}
        result_directory = output_dir / self.__class__.__name__ / self._version
        current_args = {
                arg: getattr(self, arg) for arg in inspect.signature(self.__init__).parameters.keys()
            }

        for d in result_directory.glob(f"*"):
            if not d.is_dir():
                continue

            pipeline_info = FileManager.load_json(d / "pipeline_info.json")
            pipeline_args = pipeline_info.get("arguments", {})

            if pipeline_args != current_args:
                continue

            for sub_d in d.glob("*"):
                if not sub_d.is_dir():
                    continue

                info_file = sub_d / "info.json"
                if info_file.exists():
                    info = FileManager.load_json(info_file)
                    found_info = {
                        "date": info["date"],
                        "inputs": info["inputs"],
                        "hash": sub_d.name
                    }
                    if (
                        existing_results.get(sub_d.name) is None
                        or datetime.strptime(info["date"], '%Y-%m-%d') >
                        datetime.strptime(existing_results[sub_d.name]["date"], '%Y-%m-%d')
                    ):
                        existing_results[sub_d.name] = found_info
        return existing_results

    def _are_file_hashes_identical(self, study_inputs: Dict[str, Path], existing_inputs: Dict[str, str]) -> bool:
        """Compare file hashes to determine if the inputs have changed."""
        if set(str(p) for p in study_inputs.values()) != set(existing_inputs.keys()):
            return False

        for existing_file, hash_val in existing_inputs.items():
            if FileManager.calculate_md5(Path(existing_file)) != hash_val:
                return False

        return True

    def _identify_matching_results(self, dataset: Any, existing_results: Dict[str, Dict]) -> Dict[str, bool]:
        """Compare dataset inputs with existing results."""
        dataset_inputs = self.gather_all_study_inputs(dataset)
        return {
            db_id: self._are_file_hashes_identical(study_inputs, existing_results.get(db_id, {}).get("inputs", {}))
            for db_id, study_inputs in dataset_inputs.items()
        }



class IndependentPipeline(Pipeline):
    """Pipeline that processes each study independently."""

    def process_dataset(self, dataset: Any, output_directory: Path, **kwargs):
        """Process individual studies through the pipeline independently."""
        hash_str = self.create_directory_hash(dataset)
        hash_outdir = output_directory / self.__class__.__name__ / self._version / hash_str

        # If the directory exists, find the next available directory with a suffix like "-1", "-2", etc.
        if hash_outdir.exists():
            hash_outdir = FileManager.get_next_available_dir(hash_outdir)
        hash_outdir.mkdir(parents=True, exist_ok=True)

        self.write_pipeline_info(hash_outdir)
        # Process each study individually
        filtered_dataset = self.filter_inputs(output_directory, dataset)
        for db_id, study in filtered_dataset.data.items():
            study_inputs = self.collect_study_inputs(study)
            study_outdir = hash_outdir / db_id
            study_outdir.mkdir(parents=True, exist_ok=True)

            results = self._process_inputs(study_inputs, **kwargs)
            if results is not None:  # Only save if validation succeeded
                FileManager.write_json(study_outdir / "results.json", results)
                self.write_study_info(hash_outdir, db_id, study_inputs)


class DependentPipeline(Pipeline):
    """Pipeline that processes all studies as a group."""

    def check_for_changes(self, output_directory: Path, dataset: Any) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self._filter_existing_results(output_directory, dataset)
        matching_results = self._identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        return any(not match for match in matching_results.values())

    def process_dataset(self, dataset: Any, output_directory: Path, **kwargs):
        """Process all studies through the pipeline as a group."""
        hash_str = self.create_directory_hash(dataset)
        hash_outdir = output_directory / self.__class__.__name__ / self._version / hash_str

        # Check if there are any changes for dependent mode
        if not self.check_for_changes(output_directory, dataset):
            print("No changes detected, skipping pipeline execution.")
            return  # No changes, so we skip the pipeline

        # If the directory exists, find the next available directory with a suffix like "-1", "-2", etc.
        if hash_outdir.exists():
            hash_outdir = FileManager.get_next_available_dir(hash_outdir)
        hash_outdir.mkdir(parents=True, exist_ok=True)

        self.write_pipeline_info(hash_outdir)
        # Collect all inputs and run the group function at once
        all_study_inputs = self.gather_all_study_inputs(dataset)
        grouped_results = self._process_inputs(all_study_inputs, **kwargs)
        if grouped_results is not None:  # Only process if validation succeeded
            for db_id, results in grouped_results.items():
                study_outdir = hash_outdir / db_id
                study_outdir.mkdir(parents=True, exist_ok=True)
                FileManager.write_json(study_outdir / "results.json", results)
                self.write_study_info(hash_outdir, db_id, all_study_inputs[db_id])


class BasePromptPipeline(IndependentPipeline):
    """Pipeline that uses a prompt and a pydantic schema to extract information from text."""
    
    # Class attributes to be defined by subclasses
    _prompt: str = None  # Prompt template for extraction
    _schema: Type[BaseModel] = None  # Pydantic schema for validation

    def __init__(
        self,
        extraction_model: str,
        inputs: tuple = ("text",),
        input_sources: tuple = ("pubget", "ace"),
        env_variable: Optional[str] = None,
        env_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize the prompt-based pipeline.
        
        Args:
            extraction_model: Model to use for extraction (e.g., 'gpt-4')
            inputs: Input types required
            input_sources: Valid input sources
            env_variable: Environment variable containing API key
            env_file: Path to file containing API key
            **kwargs: Additional configuration parameters
        """
        if not self._prompt:
            raise ValueError("Subclass must define _prompt template")

        super().__init__(inputs=inputs, input_sources=input_sources)
        self.extraction_model = extraction_model
        self.env_variable = env_variable
        self.env_file = env_file
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
        
        if 'gpt' in self.extraction_model.lower():
            return OpenAI(api_key=api_key)
        raise ValueError(f"Model {self.extraction_model} not supported")
    
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

    def pre_process(self, text: str) -> str:
        """Pre-process text before extraction. Override in subclass if needed.
        
        Args:
            text: Raw text to process
            
        Returns:
            Processed text
        """
        return text

    def post_process(self, predictions: dict) -> Optional[dict]:
        """Post-process and validate predictions. Override in subclass if needed.
        
        Args:
            predictions: Raw predictions from model
            
        Returns:
            Processed predictions or None if validation fails
        """
        return predictions
    
    def run(self, study_inputs: Dict[str, Path], n_cpus: int = 1) -> Dict[str, Any]:
        """Run the core extraction pipeline logic.
        
        Args:
            study_inputs: Dictionary of input file paths
            n_cpus: Number of CPUs to use
            
        Returns:
            Dictionary containing predictions and clean_predictions
        """
        # Initialize client
        client = self._load_client()
        
        # Read and pre-process text
        with open(study_inputs["text"]) as f:
            text = f.read()
        text = self.pre_process(text)

        # Create prompt configuration
        prompt_config = {
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt.replace("${text}", text) + "\n Call the extractData function to save the output."
                }
            ],
            "output_schema": self._schema.model_json_schema()
        }
        if self.kwargs:
            prompt_config.update(self.kwargs)

        # Extract predictions
        predictions = extract_from_text(
            text,
            model=self.extraction_model,
            client=client,
            **prompt_config
        )

        if not predictions:
            logging.warning("No predictions found")
            return {"predictions": None, "clean_predictions": None}
        
        # Allow additional post-processing by subclasses
        clean_predictions = self.post_process(predictions)
        
        return {
            "predictions": predictions,
            "clean_predictions": clean_predictions
        }
