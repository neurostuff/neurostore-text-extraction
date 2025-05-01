"""Base pipeline and extraction functionality.

Provides core abstractions for pipeline execution and data extraction:
- Type definitions for pipeline data and configuration
- Base classes for extractors and pipelines
- Infrastructure for I/O and execution flow
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import concurrent.futures
from datetime import datetime
from functools import reduce
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from ns_extract.dataset import Dataset, Study

# Type variables for generic classes
T = TypeVar("T")  # Generic type
ExtractorT = TypeVar("ExtractorT", bound="Extractor")  # Extractor types
PipelineT = TypeVar("PipelineT", bound="Pipeline")  # Pipeline types

# Type aliases
FilePath = Union[str, Path]  # Path inputs
PipelineConfig = Dict[str, Dict[str, str]]  # Pipeline configuration
StudyResults = Dict[str, Any]  # Single study results
PipelineResults = Dict[str, StudyResults]  # Multiple study results
ValidationResult = Tuple[bool, StudyResults]  # Validation result format

# Constants
PIPELINE_VERSION = "latest"  # Default pipeline version
CONFIG_VERSION = "latest"  # Default config version

# Input type constants
SOURCE_INPUTS = ["text", "coordinates", "metadata"]
PIPELINE_INPUTS = ["results", "raw_results", "info"]


logger = logging.getLogger(__name__)


def deep_getattr(obj: Any, attr_path: str, default: Any = None) -> Any:
    try:
        return reduce(getattr, attr_path.split("."), obj)
    except AttributeError:
        return default


class FileManager:
    """Utility class for file handling operations."""

    @staticmethod
    def calculate_md5(file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        with file_path.open("r") as f:
            file_contents = f.read()
        return hashlib.md5(file_contents.encode()).hexdigest()

    @staticmethod
    def load_json(file_path: Path) -> Dict:
        """Load JSON from a file."""
        with file_path.open("r") as f:
            return json.load(f)

    @staticmethod
    def write_json(file_path: Path, data: Dict):
        """Write JSON to a file."""
        with file_path.open("w") as f:
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


class InputManager:
    """Handles input file operations and data loading for pipelines."""

    def __init__(
        self,
        pipeline_directory: Path,
        input_sources: tuple = ("pubget", "ace"),
        pipeline_inputs: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize input manager.

        Args:
            pipeline_directory: Base directory for pipeline data
            input_sources: Sources to accept file inputs from
            pipeline_inputs: Dict mapping pipeline names to lists of inputs to use
        """
        self.pipeline_directory = pipeline_directory
        self.input_sources = input_sources
        self.pipeline_inputs = pipeline_inputs or {}

    def get_source_text(
        self, study_inputs: Dict[str, str], source_type: str = "text"
    ) -> str:
        """Get text content from source file.

        Args:
            study_inputs: Dict containing input file paths
            source_type: Type of source file to read (text, coordinates, metadata)

        Returns:
            Content of the requested file

        Raises:
            KeyError: If source_type not found in inputs
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
        """
        if source_type not in study_inputs:
            raise KeyError(
                f"Required input '{source_type}' not found in study inputs. "
                f"Available inputs: {list(study_inputs.keys())}"
            )

        try:
            with open(study_inputs[source_type], "r") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Input file not found: {study_inputs[source_type]}. "
                f"Required for input type: {source_type}"
            )
        except IOError as e:
            raise IOError(f"Error reading file {study_inputs[source_type]}: {str(e)}")

    def get_source_json(
        self, study_inputs: Dict[str, str], source_type: str = "metadata"
    ) -> Dict:
        """Get JSON content from source file.

        Args:
            study_inputs: Dict containing input file paths
            source_type: Type of source file to read (metadata, coordinates)

        Returns:
            Parsed JSON content

        Raises:
            KeyError: If source_type not found in inputs
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            IOError: If file can't be read
        """
        if source_type not in study_inputs:
            raise KeyError(
                f"Required input '{source_type}' not found in study inputs. "
                f"Available inputs: {list(study_inputs.keys())}"
            )

        try:
            with open(study_inputs[source_type], "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Input file not found: {study_inputs[source_type]}. "
                f"Required for input type: {source_type}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {study_inputs[source_type]}: {str(e)}", e.doc, e.pos
            )
        except IOError as e:
            raise IOError(f"Error reading file {study_inputs[source_type]}: {str(e)}")

    def get_pipeline_result(
        self, study_id: str, pipeline_name: str, version: str, config_hash: str
    ) -> Dict:
        """Get results from another pipeline.

        Args:
            study_id: Study ID to get results for
            pipeline_name: Name of pipeline to get results from
            version: Pipeline version
            config_hash: Pipeline config hash

        Returns:
            Pipeline results for the specified study

        Raises:
            ValueError: If results not found
        """
        result_path = (
            self.pipeline_directory
            / pipeline_name
            / version
            / config_hash
            / study_id
            / "results.json"
        )

        if not result_path.exists():
            raise ValueError(f"Missing results for pipeline {pipeline_name}")

        return FileManager.load_json(result_path)

    def get_all_inputs(
        self,
        study_id: str,
        source_inputs: Dict[str, str],
        pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Get all inputs needed for a study.

        Args:
            study_id: Study ID
            source_inputs: Dict mapping input types to file paths
            pipeline_kwargs: Optional kwargs for getting pipeline results

        Returns:
            Dict containing all required inputs
        """
        inputs = {}

        # Get source file inputs
        for input_type, path in source_inputs.items():
            if input_type in ("text", "coordinates"):
                inputs[input_type] = self.get_source_text(source_inputs, input_type)
            elif input_type == "metadata":
                inputs[input_type] = self.get_source_json(source_inputs, input_type)

        # Get pipeline inputs if needed
        if pipeline_kwargs:
            for name, kwargs in pipeline_kwargs.items():
                result = self.get_pipeline_result(
                    study_id=study_id,
                    pipeline_name=name,
                    version=kwargs["version"],
                    config_hash=kwargs["config_hash"],
                )
                # Add with pipeline name prefix to avoid collisions
                for result_type in self.pipeline_inputs.get(name, []):
                    inputs[f"{name}.{result_type}"] = result.get(result_type)

        return inputs

    def collect_dataset_inputs(
        self, dataset: Dataset, input_pipeline_kwargs: Optional[PipelineConfig] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Collect all required inputs for dataset processing.

        This method coordinates input collection across all studies,
        gathering both source file inputs and pipeline dependencies.

        Args:
            dataset: Dataset to collect inputs for
            input_pipeline_kwargs: Optional pipeline configuration arguments
                Example:
                    {
                        "pipeline_name":
                            {
                                "pipeline_dir": "/some/path",
                                "version": "1.0.0",
                                "config_hash": "abc123"
                            }
                    }

        Returns:
            Dict mapping study IDs to their collected inputs
        """
        return {
            study_id: self._collect_study_inputs(study, input_pipeline_kwargs)
            for study_id, study in dataset.data.items()
        }

    def _collect_study_inputs(
        self, study: Study, input_pipeline_kwargs: Optional[PipelineConfig] = None
    ) -> Dict[str, Any]:
        """Collect all required inputs for a single study.

        Args:
            study: Study to collect inputs for
            input_pipeline_kwargs: Optional pipeline configuration arguments

        Returns:
            Dict of collected study inputs
        """
        inputs = {}

        # Get source file inputs
        for source_type in SOURCE_INPUTS:
            if hasattr(study, source_type):
                inputs[source_type] = getattr(study, source_type)

        # Get pipeline dependency inputs
        if input_pipeline_kwargs:
            for name in input_pipeline_kwargs:
                result_path = study.pipeline_results[name].result
                inputs[name] = result_path

        return inputs


class OutputManager:
    """Handles output file operations and results writing for pipelines.

    Responsible for:
    - Writing pipeline results and metadata
    - Managing output directory structure
    - Handling file I/O operations
    """

    def __init__(self, extractor_name: str, version: str, config_hash: str) -> None:
        """Initialize output manager.

        Args:
            extractor_name: Name of the extractor class
            version: Pipeline version
            config_hash: Configuration hash
        """
        self.extractor_name = extractor_name
        self.version = version
        self.config_hash = config_hash

    def _validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate that results have the expected structure.

        Valid results must be either:
        1. A direct results dict
        2. A dict with 'results' key and optional 'raw_results' key

        Args:
            results: Results dict to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(results, dict):
            return False

        # Case 1: Direct results dict
        if all(
            isinstance(v, (dict, list, str, int, float, bool)) for v in results.values()
        ):
            return True

        # Case 2: Results with optional raw_results
        if "results" in results:
            if not isinstance(results["results"], dict):
                return False
            if "raw_results" in results and not isinstance(
                results["raw_results"], dict
            ):
                return False
            return True

        return False

    def write_pipeline_info(self, output_dir: Path) -> "OutputManager":
        """Write pipeline metadata to pipeline_info.json.

        Args:
            output_dir: Directory to write pipeline info

        Returns:
            Self for method chaining

        Raises:
            ValueError: If output_dir is not a Path
            IOError: If writing fails
        """
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir must be a Path object")

        info = {
            "date": datetime.now().isoformat(),
            "version": self.version,
            "config_hash": self.config_hash,
            "extractor": self.extractor_name,
        }

        try:
            info_path = output_dir / "pipeline_info.json"
            FileManager.write_json(info_path, info)
        except IOError as e:
            raise IOError(f"Failed to write pipeline info: {str(e)}")

        return self

    def write_study_results(
        self, study_dir: Path, study_id: str, results: Dict[str, Any]
    ) -> "OutputManager":
        """Write study results to directory.

        Args:
            study_dir: Directory for study files
            study_id: ID of the study
            results: Study results to write

        Returns:
            Self for method chaining

        Raises:
            IOError: If writing fails
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not isinstance(study_dir, Path):
            raise ValueError("study_dir must be a Path object")
        if not study_id:
            raise ValueError("study_id cannot be empty")
        if not self._validate_results(results):
            raise ValueError(
                "Invalid results structure. Must be either a results dict "
                "or contain 'results' key with optional 'raw_results'"
            )

        try:
            study_dir.mkdir(exist_ok=True)

            # Write raw results if provided
            if "raw_results" in results:
                raw_path = study_dir / "raw_results.json"
                FileManager.write_json(raw_path, results["raw_results"])

            # Write final results
            results_path = study_dir / "results.json"
            final_results = results.get("results", results)
            FileManager.write_json(results_path, final_results)

        except IOError as e:
            raise IOError(f"Failed to write results for study {study_id}: {str(e)}")

        return self


class Pipeline:
    """Base pipeline class for handling I/O operations.

    data_pond_inputs: Inputs for the data pond structured like this:
        {("pubget", "ace"): ("text", "coordinates")}
        where the first tuple is the input source and is ordered so
        the first element will be prioritized over the second.
        If a pubget source is not available, the second element will be used.
        the values are the input types that will be used for the extractor.
    pipeline_inputs: Inputs for the pipeline structured like this:
        {("participant_info"): ("results", "raw_results")}
    """

    _data_pond_inputs: Dict[str, str] = Field(default_factory=dict)
    _pipeline_inputs: Dict[str, str] = Field(default_factory=dict)

    def __init__(
        self,
        extractor: Extractor,
    ):
        """Initialize pipeline.

        Args:
            extractor: Extractor instance that defines the data transformation
        """
        self.extractor = extractor
        self.input_manager = None
        self.output_manager = None

    def _create_hash_string(self, dataset: Dataset) -> str:
        """Create string for hashing pipeline configuration.

        The hash combines:
        - Dataset study IDs (sorted)
        - Extractor configuration
        - Pipeline version and config
        """
        # Dataset study IDs in sorted order
        dataset_str = "_".join(sorted(dataset.data.keys()))

        # Extractor config - exclude private attrs
        transform_str = "_".join(
            [
                f"{key}_{val}"
                for key, val in self.extractor.__dict__.items()
                if not key.startswith("_")
            ]
        )

        # Pipeline config
        pipeline_str = f"{self.extractor._version}_{self._config_hash}"

        return f"{dataset_str}_{transform_str}_{pipeline_str}"

    def _create_output_directory(self, dataset: Dataset, base_dir: Path) -> Path:
        """Create uniquely hashed output directory.

        Directory structure:
        base_dir/
          extractor_name/
            version/
              hash/
        """
        # Create hash from configuration
        hash_str = hashlib.shake_256(
            self._create_hash_string(dataset).encode()
        ).hexdigest(6)

        # Create directory path
        outdir = base_dir / self.extractor.__class__.__name__ / self.extractor._version / hash_str

        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def _prepare_pipeline(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Tuple[Path, Path]:
        """Prepare pipeline execution by setting up dependencies and creating output directory.

        Args:
            dataset: Dataset to process
            output_directory: Directory to write outputs
            input_pipeline_kwargs: Optional kwargs for input pipelines

        Returns:
            hash_output_directory as Path
        """
        output_directory = Path(output_directory)

        # Set up dependencies from other pipeline outputs
        if input_pipeline_kwargs:
            for name, dependency_info in input_pipeline_kwargs.items():
                dataset.add_pipeline(
                    pipeline_name=name,
                    pipeline_dir=dependency_info["pipeline_dir"],
                    version=dependency_info.get("version", PIPELINE_VERSION),
                    config_hash=dependency_info.get("config_hash", CONFIG_VERSION),
                )

        # Create output directory with hash
        hash_outdir = self._create_output_directory(dataset, output_directory)

        return hash_outdir

    @abstractmethod
    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Process a dataset through the pipeline.

        Args:
            dataset: Dataset to process
            output_directory: Directory to write outputs
            input_pipeline_kwargs: Optional kwargs for input pipelines
            **kwargs: Additional arguments passed to extractor
        """
        pass

    def _create_directory_hash(self, dataset: Dataset) -> str:
        """Create hash string for output directory."""
        # Hash based on dataset study IDs and extractor args
        dataset_str = "_".join(sorted(dataset.data.keys()))
        extractor_str = self.extractor.get_config_string()
        full_str = f"{dataset_str}_{extractor_str}"
        return hashlib.shake_256(full_str.encode()).hexdigest(6)

    def _write_results(self, output_dir: Path, results: Dict[str, Any]) -> None:
        """Write pipeline results and metadata.

        Handles writing both raw and processed results when available.
        If extractor's post_process modifies results, both versions are written:
        - raw_results.json: Original transform output
        - results.json: Post-processed output

        Args:
            output_dir: Directory to write results to
            results: Results from extractor to write

        Raises:
            IOError: If writing any file fails
            ValueError: If output_dir is not a Path
        """
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir must be a Path object")

        # Create manager and write pipeline info
        manager = OutputManager(
            extractor_name=self.extractor.__class__.__name__,
            version=self._version,
            config_hash=self._config_hash,
        )

        # Write pipeline info first
        manager.write_pipeline_info(output_dir)

        # Write each study's results
        for study_id, study_results in results.items():
            study_dir = output_dir / study_id
            study_dir.mkdir(exist_ok=True, parents=True)
            manager.write_study_results(study_dir, study_id, study_results)


class IndependentPipeline(Pipeline):
    """Pipeline that processes each study independently."""

    def _process_and_write_study(
        self, study_data: Tuple[str, Dict[str, Any], Path], hash_outdir: Path, **kwargs
    ) -> bool:
        """Process a single study and write its results.

        Args:
            study_data: Tuple of (db_id, study_inputs, study_outdir)
            hash_outdir: Base output directory
            **kwargs: Additional arguments for processing

        Returns:
            bool: True if processing was successful
        """
        db_id, study_inputs, study_outdir = study_data
        study_outdir.mkdir(parents=True, exist_ok=True)

        # Process the inputs with study ID for error tracking
        outputs = self._process_inputs(study_inputs, study_id=db_id, **kwargs)

        if outputs:
            # Write results immediately
            for output_type, output in outputs.items():
                if output_type == "results":
                    is_valid, output = output
                    self.write_study_info(hash_outdir, db_id, study_inputs, is_valid)
                FileManager.write_json(study_outdir / f"{output_type}.json", output)
            return True
        return False

    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
        num_workers=1,
        **kwargs,
    ):
        """Process individual studies through the pipeline independently."""
        hash_outdir = self._prepare_pipeline(
            dataset, output_directory, input_pipeline_kwargs
        )

        if not hash_outdir.exists():
            hash_outdir.mkdir(parents=True)
            # Include transform arguments in pipeline info
            transform_kwargs = {"num_workers": num_workers, **kwargs}
            self.write_pipeline_info(
                hash_outdir,
                input_pipeline_kwargs=input_pipeline_kwargs,
                transform_kwargs=transform_kwargs,
            )

        # Filter and prepare studies that need processing
        filtered_dataset = self.filter_inputs(output_directory, dataset)
        studies_to_process = []

        for db_id, study in filtered_dataset.data.items():
            study_inputs = self.collect_study_inputs(study, input_pipeline_kwargs)
            study_outdir = hash_outdir / db_id

            # Skip if already processed
            if study_outdir.exists():
                continue

            studies_to_process.append((db_id, study_inputs, study_outdir))

        if not studies_to_process:
            print("No studies need processing")
            return

        # Process and write results as they complete
        print(f"Processing {len(studies_to_process)} studies...")
        success_count = 0

        with tqdm(total=len(studies_to_process), desc="Processing studies") as pbar:
            if num_workers > 1:
                # Parallel processing
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as exc:
                    futures = []
                    for study_data in studies_to_process:
                        future = exc.submit(
                            self._process_and_write_study,
                            study_data,
                            hash_outdir,
                            **kwargs,
                        )
                        future.add_done_callback(lambda _: pbar.update(1))
                        futures.append(future)

                    # Wait for completion and count successes
                    for future in concurrent.futures.as_completed(futures):
                        if future.result():
                            success_count += 1
            else:
                # Serial processing
                for study_data in studies_to_process:
                    if self._process_and_write_study(study_data, hash_outdir, **kwargs):
                        success_count += 1
                    pbar.update(1)

        print(
            f"Completed processing {success_count} "
            f"of {len(studies_to_process)} studies successfully"
        )


class DependentPipeline(Pipeline):
    """Pipeline that processes all studies as a group."""

    def _create_hash_string(self, dataset: Dataset) -> str:
        """Create hash string including dataset keys for group processing.

        For dependent pipelines, we include dataset keys since all studies are
        processed together. This extends the base pipeline hash with dataset keys.
        """
        base_hash = super()._create_hash_string(dataset)
        dataset_str = self._serialize_dataset_keys(dataset)
        return f"{dataset_str}_{base_hash}"

    def check_for_changes(
        self,
        output_directory: Path,
        dataset: Dataset,
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self._filter_existing_results(output_directory, dataset)
        matching_results = self._identify_matching_results(
            dataset, existing_results, input_pipeline_kwargs=input_pipeline_kwargs
        )
        # Return True if any of the studies' inputs have changed or if new studies exist
        return any(not match for match in matching_results.values())

    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Process all studies through the pipeline as a group."""
        hash_outdir = self._prepare_pipeline(
            dataset, output_directory, input_pipeline_kwargs
        )

        # Check if there are any changes for dependent mode
        if not self.check_for_changes(output_directory, dataset, input_pipeline_kwargs):
            logger.info("No changes detected, skipping pipeline execution.")
            return  # No changes, so we skip the pipeline

        # If the directory exists, find the next available directory
        # with a suffix like "-1", "-2", etc.
        if hash_outdir.exists():
            hash_outdir = FileManager.get_next_available_dir(hash_outdir)
        hash_outdir.mkdir(parents=True, exist_ok=True)
        self.write_pipeline_info(
            hash_outdir,
            input_pipeline_kwargs=input_pipeline_kwargs,
            transform_kwargs=kwargs,
        )

        # Collect all inputs and run the group function at once
        all_study_inputs = {
            db_id: self.collect_study_inputs(study, input_pipeline_kwargs)
            for db_id, study in dataset.data.items()
        }

        grouped_outputs = self._process_inputs(
            all_study_inputs, study_id="grouped", **kwargs
        )
        if grouped_outputs:
            for output_type, outputs in grouped_outputs.items():
                if outputs is not None:
                    for db_id, _output in outputs.items():
                        study_outdir = hash_outdir / db_id
                        study_outdir.mkdir(parents=True, exist_ok=True)
                        if output_type == "results":
                            is_valid, _output = _output
                            self.write_study_info(
                                hash_outdir, db_id, all_study_inputs[db_id], is_valid
                            )
                        FileManager.write_json(
                            study_outdir / f"{output_type}.json", _output
                        )

    def validate_results(self, results, **kwargs):
        """Apply validation to each studys results in the grouped pipeline."""
        validated_results = {}
        for db_id, study_results in results.items():
            study_results = super().validate_results(study_results, **kwargs)
            validated_results[db_id] = study_results
        return validated_results


class Extractor(ABC):
    """Base class for data transformation logic.

    Handles only the transformation of input data to output data.
    All I/O operations and pipeline coordination are handled by the Pipeline class.

    Required Methods:
        transform: Implement the core data transformation logic

    Optional Methods:
        post_process: Override to modify results before validation
    """

    _version = None
    _output_schema: Type[BaseModel] = None

    def __init__(self, **kwargs):
        """Initialize extractor and verify schema exists."""
        if not self._output_schema:
            raise ValueError("Subclass must define _output_schema class variable")
        if not self._version:
            raise ValueError("Subclass must define _version class variable")

        if isinstance(self, IndependentPipeline):
            IndependentPipeline.__init__(self, extractor=self)
        if isinstance(self, DependentPipeline):
            DependentPipeline.__init__(self, extractor=self)

    @abstractmethod
    def _transform(self, inputs: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Transform input data into output format.

        Args:
            inputs: Dict mapping study IDs to their input data
            **kwargs: Additional transformation arguments

        Returns:
            Dict mapping study IDs to their transformed outputs
        """
        pass

    def transform(self, inputs: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Transform input data into output format.

        Args:
            inputs: Dict mapping study IDs to their input data
            **kwargs: Additional transformation arguments

        Returns:
            Dict mapping study IDs to their transformed outputs
        """
        results = self._transform(inputs, **kwargs)
        cleaned_results = self.post_process(results)
        if cleaned_results is None:
            cleaned_results = results
        valid = self.validate_output(cleaned_results)

        return (cleaned_results, results, valid)

    def validate_output(self, output: Dict) -> Tuple[bool, Dict]:
        """Validate output against schema.

        Args:
            output: Data to validate against schema

        Returns:
            Tuple of (is_valid, validated_output)
        """
        try:
            self._output_schema.model_validate(output)
            return True
        except Exception as e:
            logging.error(f"Output validation error: {str(e)}")
            return False

    def post_process(self, results: Dict) -> Dict:
        """Optional hook for post-processing transform results.

        Override in subclasses to modify results before validation.

        Args:
            results: Raw transform results

        Returns:
            Post-processed results, or None if no changes are made
        """
        return None
