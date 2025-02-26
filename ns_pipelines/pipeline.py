from datetime import datetime
import inspect
import json
import hashlib
import os
import logging
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Type, Tuple

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
    _output_schema: Type[BaseModel] = None  # Required schema for output validation

    def __init__(self, inputs: Union[tuple, list] = ("text",), input_sources: tuple = ("pubget", "ace")):
        if not self._output_schema:
            raise ValueError("Subclass must define _output_schema class")
            
        self.inputs = inputs
        self.input_sources = input_sources
        self._pipeline_type = inspect.getmro(self.__class__)[1].__name__.lower().rstrip("pipeline")

    @abstractmethod
    def transform_dataset(self, dataset: Any, output_directory: Path, **kwargs):
        """Process a full dataset through the pipeline."""
        pass

    def _process_inputs(self, study_inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process inputs through the full pipeline flow: pre-process, execute, post-process, validate.
        
        Returns a dict with:
            - results: Validated results
            - raw_results: Raw results if post-processing was applied
        """
        try:
            # Pre-process inputs
            processed_inputs = self.pre_process(study_inputs)
            if not processed_inputs:
                logging.error("Pre-processing returned no inputs")
                return None

            # Execute core pipeline logic
            raw_results = self.execute(processed_inputs, **kwargs)
            if not raw_results:
                return None

            # Post-process results
            try:
                post_results = self.post_process(raw_results)
            except Exception as e:
                logging.error(f"Post-processing failed: {e}")
                post_results = None

            results = self.validate_results(
                post_results or raw_results
            )

            output = {
                "results": results,
            }

            if post_results:
                output["raw_results"] = raw_results

            return output


        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            return None

    def validate_results(self, results: dict) -> Optional[dict]:
        """Validate results against the output schema.
        
        Args:
            results: Raw or post-processed results from pipeline
            
        Returns:
            Validated results or None if validation fails
        """
        try:
            validated = self._output_schema.model_validate(results)
            return True, validated.model_dump()
        except Exception as e:
            logging.error(f"Raw result validation error: {e}")
            return False, results

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-process inputs before pipeline execution. Override in subclass if needed.
        
        Args:
            inputs: Raw inputs to the pipeline
            
        Returns:
            Processed inputs for pipeline execution
        """
        return inputs
    
    def post_process(self, results: dict) -> dict:
        """Post-process results before validation. Override in subclass if needed.
        
        Args:
            results: Raw results from pipeline
            
        Returns:
            Processed results or None if processing fails
        """
        pass

    @abstractmethod
    def execute(self, processed_inputs: Dict[str, Any], **kwargs) -> Dict:
        """Execute the core pipeline logic using pre-processed inputs.
        
        Args:
            processed_inputs: Pre-processed inputs ready for pipeline execution
            **kwargs: Additional arguments for pipeline execution
            
        Returns:
            Raw results from pipeline execution
        """
        pass

    def create_directory_hash(self, dataset: Any, output_directory: Path) -> Tuple[Path, str]:
        """Create a hash for the dataset."""
        dataset_str = self._serialize_dataset_keys(dataset)
        arg_str = self._serialize_pipeline_args()
        hash_str = hashlib.shake_256(f"{dataset_str}_{arg_str}".encode()).hexdigest(6)
        outdir = output_directory / self.__class__.__name__ / self._version / hash_str
        return outdir, hash_str

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

    def write_study_info(self, hash_outdir: Path, db_id: str, study_inputs: Dict[str, Path], is_valid: bool):
        """Write information about the current run to an info.json file."""
        output_info = {
            "date": datetime.now().isoformat(),
            "inputs": {str(input_file): FileManager.calculate_md5(input_file) for input_file in study_inputs.values()},
            "valid": is_valid,
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
        
        current_args = json.loads(json.dumps(current_args))

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

    def transform_dataset(self, dataset: Any, output_directory: Path, **kwargs):
        """Process individual studies through the pipeline independently."""
        hash_outdir, hash_str = self.create_directory_hash(dataset, output_directory)

        if not hash_outdir.exists():
            hash_outdir.mkdir(parents=True)
            self.write_pipeline_info(hash_outdir)

        # Process each study individually
        filtered_dataset = self.filter_inputs(output_directory, dataset)
        for db_id, study in filtered_dataset.data.items():
            study_inputs = self.collect_study_inputs(study)
            study_outdir = hash_outdir / db_id

            # If directory exists, this study has already been processed
            if study_outdir.exists():
                continue

            study_outdir.mkdir(parents=True)

            outputs = self._process_inputs(study_inputs, **kwargs)
            if outputs:
                for output_type, output in outputs.items():
                    if output_type == "results":
                        is_valid, output = output
                        self.write_study_info(hash_outdir, db_id, study_inputs, is_valid)
                    FileManager.write_json(study_outdir / f"{output_type}.json", output)

class DependentPipeline(Pipeline):
    """Pipeline that processes all studies as a group."""

    def check_for_changes(self, output_directory: Path, dataset: Any) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self._filter_existing_results(output_directory, dataset)
        matching_results = self._identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        return any(not match for match in matching_results.values())

    def transform_dataset(self, dataset: Any, output_directory: Path, **kwargs):
        """Process all studies through the pipeline as a group."""
        hash_outdir, hash_str = self.create_directory_hash(dataset, output_directory)

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
        grouped_outputs = self._process_inputs(all_study_inputs, **kwargs)
        if grouped_outputs:
            for output_type, outputs in grouped_outputs.items():
                if outputs is not None:
                    for db_id, _output in outputs.items():
                        study_outdir = hash_outdir / db_id
                        study_outdir.mkdir(parents=True, exist_ok=True)
                        if output_type == "results":
                            is_valid, _output = _output
                            self.write_study_info(hash_outdir, db_id, all_study_inputs[db_id], is_valid)
                        FileManager.write_json(study_outdir / f"{output_type}.json", _output)


    def validate_results(self, results):
        """ Apply validation to each study's results in the grouped pipeline."""
        validated_results = {}
        for db_id, study_results in results.items():
            study_results = super().validate_results(study_results)
            validated_results[db_id] = study_results
        return validated_results


class BasePromptPipeline(IndependentPipeline):
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
        if not self._extraction_schema:
            self._extraction_schema = self._output_schema

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

        # Create prompt configuration
        prompt_config = {
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt.replace("${text}", text) + "\n Call the extractData function to save the output."
                }
            ],
            "output_schema": self._extraction_schema.model_json_schema()
        }
        if self.kwargs:
            prompt_config.update(self.kwargs)

        # Extract predictions
        results = extract_from_text(
            text,
            model=self.extraction_model,
            client=client,
            **prompt_config
        )

        if not results:
            logging.warning("No results found")
            return None
            
        return results
