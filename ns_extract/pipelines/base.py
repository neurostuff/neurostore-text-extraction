"""Base pipeline and extraction functionality.

This module provides core abstractions for pipeline execution and data extraction:
- Type definitions and protocols for pipeline data and configuration
- Base classes for extractors and pipelines
- Infrastructure for I/O and execution flow
- Validation and error handling

The module implements a flexible pipeline architecture that supports both:
- Independent processing (each study processed separately)
- Dependent processing (studies processed as a group)

Key Components:
- Pipeline: Abstract base for handling I/O operations
- Extractor: Base class for implementing data transformation logic
- Validation: Schema-based validation of pipeline outputs
- Error Handling: Structured exception handling and recovery
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import concurrent.futures
from datetime import datetime
from functools import reduce
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel
from tqdm.auto import tqdm
from ns_extract.dataset import Dataset
from ns_extract.pipelines.utils import (
    StudyInputsMixin,
    PipelineOutputsMixin,
)
from ns_extract.pipelines.exceptions import (
    FileOperationError,
    InputError,
    ProcessingError,
    ValidationError,
)
from ns_extract.pipelines.data_structures import (
    InputPipelineInfo,
    NORMALIZE_TEXT,
    EXPAND_ABBREVIATIONS,
)


from ns_extract.pipelines.normalize import (
    normalize_string,
    load_abbreviations,
    resolve_abbreviations,
)


# Type aliases with documentation
PipelineConfig = Dict[str, Dict[str, str]]  # Pipeline configuration mapping
StudyResults = Dict[str, Any]  # Results for a single study
PipelineResults = Dict[str, StudyResults]  # Results across multiple studies
ValidationResult = Tuple[bool, StudyResults]  # Validation result with data
PipelineInputs = Dict[Tuple[str, ...], Tuple[str, ...]]  # Input source -> type mapping

# Input pipeline constants
PIPELINE_VERSION = "latest"  # Default pipeline version
CONFIG_VERSION = "latest"  # Default config version

# Input type constants
SOURCE_INPUTS = ["text", "coordinates", "metadata"]
PIPELINE_INPUTS = ["results", "raw_results", "info"]


logger = logging.getLogger(__name__)


def get_nested_attribute(obj: Any, attr_path: str, default: Any = None) -> Any:
    try:
        return reduce(getattr, attr_path.split("."), obj)
    except AttributeError:
        return default


class Pipeline(StudyInputsMixin, PipelineOutputsMixin):
    """Base pipeline class for handling I/O operations.

    data_pond_inputs: Inputs for the data pond structured like this:
        {("pubget", "ace"): ("text", "coordinates")}
        where the first tuple is the input source and is ordered so
        the first element will be prioritized over the second.
        If a pubget source is not available, the second element will be used.
        the values are the input types that will be used for the extractor.
    input_pipelines: Inputs for the pipeline structured like this:
        {("participant_info"): ("results", "raw_results")}
    """

    _data_pond_inputs = {}
    _input_pipelines = {}

    def __init__(
        self,
        extractor: Extractor,
    ):
        """Initialize pipeline.

        Args:
            extractor: Extractor instance that defines the data transformation
        """
        self.extractor = extractor

    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[Dict[str, InputPipelineInfo]] = None,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        """Process a dataset through the pipeline.

        Args:
            dataset: Dataset to process
            output_directory: Directory to write outputs
            input_pipeline_info: Optional configuration for input pipelines
            num_workers: Number of parallel workers (only used by IndependentPipeline)
            **kwargs: Additional arguments passed to extractor

        Raises:
            FileOperationError: If directory creation or file operations fail
            ProcessingError: If pipeline processing fails
            ValidationError: If results fail schema validation
        """
        # Get initial output directory path without creating it
        hash_outdir = self.__prepare_pipeline(
            dataset, output_directory, input_pipeline_info
        )

        # Create pipeline info object
        pipeline_info = self._create_pipeline_info(
            hash_outdir,
            kwargs,
            input_pipelines=input_pipeline_info,
        )

        # If directory exists, check for changes
        if hash_outdir.exists():
            if not self._has_study_changes(hash_outdir, dataset):
                print("No studies need processing")
                return
            # Get next available directory for new results if a DependentPipeline
            # For IndependentPipeline, we overwrite the existing directory
            if isinstance(self, DependentPipeline):
                hash_outdir = self._find_unique_directory(hash_outdir)

        # Create directory and write pipeline info
        hash_outdir.mkdir(parents=True, exist_ok=True)
        self._write_pipeline_info(hash_outdir, pipeline_info)

        # Collect and process study inputs
        try:
            self._process_dataset(
                dataset,
                hash_outdir,
                num_workers=num_workers,
                **kwargs,
            )
        except (
            InputError,
            ProcessingError,
            ValidationError,
            FileOperationError,
        ) as e:
            self.__handle_processing_error(e, hash_outdir)
            raise

        return hash_outdir

    def __prepare_pipeline(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Path:
        """Prepare pipeline execution by setting up dependencies and creating output directory.

        Args:
            dataset: Dataset to process
            output_directory: Directory to write outputs
            input_pipeline_info: Optional kwargs for input pipelines

        Returns:
            hash_output_directory as Path
        """
        output_directory = Path(output_directory)

        # Set up dependencies from other pipeline outputs
        if input_pipeline_info:
            for name, dependency_info in input_pipeline_info.items():
                dataset.add_pipeline(
                    pipeline_name=name,
                    pipeline_dir=dependency_info["pipeline_dir"],
                    version=dependency_info.get("version", PIPELINE_VERSION),
                    config_hash=dependency_info.get("config_hash", CONFIG_VERSION),
                )

        # Create hashed output directory path
        # Create hash from configuration
        hash_str = hashlib.shake_256(
            self._generate_config_hash_string(dataset).encode()
        ).hexdigest(6)

        # Create full directory path
        hash_outdir = (
            Path(output_directory)
            / self.extractor.__class__.__name__
            / self.extractor._version
            / hash_str
        )

        return hash_outdir

    def _generate_config_hash_string(self, dataset: Dataset) -> str:
        """Create string for hashing pipeline configuration.

        The hash combines:
        - Extractor configuration
        - Pipeline version and config

        Note: Dependent pipelines include dataset keys in the hash
        """

        # Extractor config - exclude private attrs
        args = list(inspect.signature(self.__init__).parameters.keys())
        args_str = "_".join([f"{arg}_{str(getattr(self, arg))}" for arg in args])
        version_str = self.extractor._version

        return f"{version_str}_{args_str}"

    def _has_study_changes(
        self,
        hash_outdir: Path,
        dataset: Dataset,
    ) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self.__filter_existing_results(hash_outdir, dataset)
        matching_results = self.__identify_matching_results(
            dataset,
            existing_results,
        )
        # Return True if any of the studies' inputs have changed or if new studies exist
        return any(not match for match in matching_results.values())

    def __filter_existing_results(
        self, hash_outdir: Path, dataset: Dataset
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Find the most recent result for an existing study.

        Args:
            hash_outdir: Output directory containing study results
            dataset: Dataset to check for existing results

        Returns:
            Dict mapping study IDs to their most recent result info
            Format: {
                "study_id": {
                    "date": "YYYY-MM-DD",
                    "inputs": {"file_path": "hash_value"},
                    "hash": "result_hash"
                }
            }
        """
        existing_results = {}
        result_directory = hash_outdir.parent
        current_args = {
            arg: getattr(self, arg)
            for arg in inspect.signature(self.__init__).parameters.keys()
        }

        current_args = json.loads(json.dumps(current_args))

        for d in result_directory.glob("*"):
            if not d.is_dir():
                continue

            pipeline_info = self._load_json(d / "pipeline_info.json")
            pipeline_args = pipeline_info.get("extractor_kwargs", {})

            if pipeline_args != current_args:
                continue

            for sub_d in d.glob("*"):
                if not sub_d.is_dir():
                    continue

                info_file = sub_d / "info.json"
                if info_file.exists():
                    info = self._load_json(info_file)
                    found_info = {
                        "date": info["date"],
                        "inputs": info["inputs"],
                        "hash": sub_d.name,
                    }
                    if existing_results.get(sub_d.name) is None or datetime.strptime(
                        info["date"], "%Y-%m-%d"
                    ) > datetime.strptime(
                        existing_results[sub_d.name]["date"], "%Y-%m-%d"
                    ):
                        existing_results[sub_d.name] = found_info
        return existing_results

    def __identify_matching_results(
        self,
        dataset: Dataset,
        existing_results: Dict[str, Dict[str, Dict[str, str]]],
    ) -> Dict[str, bool]:
        """Compare dataset inputs with existing results to identify changes.

        Args:
            dataset: Dataset to check for changes
            existing_results: Previously processed results to compare against
                Format matches return type of _filter_existing_results

        Returns:
            Dict mapping study IDs to boolean indicating if inputs match
            existing results (True = no changes, False = changes detected)
        """
        result_matches = {}
        study_inputs = self._collect_all_study_inputs(dataset)

        for db_id, study in dataset.data.items():
            # Get existing input file hashes for this study
            existing = existing_results.get(db_id, {}).get("inputs", {})

            # Skip if no existing results or no current inputs
            if not existing or db_id not in study_inputs:
                result_matches[db_id] = False
                continue

            # Use __are_file_hashes_identical to compare hashes
            result_matches[db_id] = self.__do_file_hashes_match(
                study_inputs[db_id], existing
            )

        return result_matches

    def _collect_all_study_inputs(self, dataset: Dataset) -> Dict[str, Dict[str, Path]]:
        """Collect all input file paths for each study in the dataset.

        Args:
            dataset: Dataset containing study data

        Returns:
            Dict mapping study IDs to their input files.
            Format: {
                "study_id": {
                    "input_type": Path("path/to/input/file")
                }
            }
        """
        return {
            db_id: self._collect_study_inputs(study)
            for db_id, study in dataset.data.items()
        }

    def _collect_study_inputs(self, study: Any) -> Dict[str, Path]:
        """Collect inputs for a study."""
        study_inputs = {}

        # Iterate through sources in priority order
        for sources, input_types in self._data_pond_inputs.items():
            for source in sources:
                source_obj = getattr(study, source, None)
                if source_obj:
                    for input_type in input_types:
                        input_obj = get_nested_attribute(source_obj, input_type, None)
                        if input_obj and input_type not in study_inputs:
                            study_inputs[input_type] = input_obj
                    break  # Stop after finding first valid source

        # Add pipeline inputs if needed
        if self._input_pipelines:
            for sources, input_types in self._input_pipelines.items():
                for source in sources:
                    source_obj = study.pipeline_results.get(source, None)
                    if source_obj:
                        for input_type in input_types:
                            input_obj = get_nested_attribute(
                                source_obj, input_type, None
                            )
                            if input_obj and input_type not in study_inputs:
                                # use source instead of input_type
                                # because input_type from a pipeline
                                # will always be "results"
                                # will revisit this assumption later
                                study_inputs[source] = input_obj
                        break  # Stop after finding first valid source

        return study_inputs

    def __do_file_hashes_match(
        self, study_inputs: Dict[str, Path], existing_inputs: Dict[str, str]
    ) -> bool:
        """Compare file hashes to determine if the inputs have changed."""
        if set(str(p) for p in study_inputs.values()) != set(existing_inputs.keys()):
            return False

        for existing_file, hash_val in existing_inputs.items():
            if self._calculate_md5(Path(existing_file)) != hash_val:
                return False

        return True

    @abstractmethod
    def _process_dataset(
        self,
        dataset: Dataset,
        hash_outdir: Path,
        **kwargs: Any,
    ) -> None:
        """Process the dataset.

        This method should be implemented by subclasses to handle the actual
        processing of studies according to their specific requirements
        (independent vs dependent processing).
        """
        pass

    def __handle_processing_error(self, error: Exception, hash_outdir: Path) -> None:
        """Handle errors during processing."""
        logger.error(str(error))
        if isinstance(error, FileOperationError):
            try:
                import shutil

                if hash_outdir.exists():
                    shutil.rmtree(hash_outdir)
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup after error: {str(cleanup_error)}")

    def _filter_unprocessed_studies(
        self, hash_outdir: Path, dataset: Dataset
    ) -> Dataset:
        """Filter dataset to only include studies needing processing.

        A study needs processing if:
        - It has no existing results
        - Its input files have changed since last processing
        - It's new to the dataset

        Args:
            hash_outdir: Output directory containing previous results
            dataset: Dataset to filter

        Returns:
            New Dataset containing only studies needing processing
        """
        existing_results = self.__filter_existing_results(hash_outdir, dataset)
        matching_results = self.__identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        keep_ids = set(dataset.data.keys()) - {
            db_id for db_id, match in matching_results.items() if match
        }
        return dataset.slice(keep_ids)


# Concrete Pipeline implementations
class DependentPipeline(Pipeline):
    """Pipeline that processes all studies as a group."""

    def _process_dataset(
        self,
        dataset: Dataset,
        hash_outdir: Path,
        **kwargs: Any,
    ) -> None:
        """Process all studies in the dataset as a group.

        Args:
            dataset: Dataset to process
            hash_outdir: Output directory for results
            **kwargs: Additional arguments passed to extractor

        Raises:
            InputError: If loading study inputs fails
            ProcessingError: If transformation fails
            FileOperationError: If writing results fails
            ValidationError: If results fail schema validation
        """
        try:
            # Collect all study inputs
            all_study_inputs = {
                db_id: self._collect_study_inputs(study)
                for db_id, study in dataset.data.items()
            }

            # Load all study inputs
            loaded_study_inputs = {}
            for db_id, study_inputs in all_study_inputs.items():
                try:
                    loaded_study_inputs[db_id] = self._load_study_inputs(study_inputs)
                except (IOError, ValueError) as e:
                    raise InputError(
                        f"Failed to load inputs for study {db_id}: {str(e)}"
                    )

            # Process loaded inputs and get results
            transform_outputs = self.transform(loaded_study_inputs, **kwargs)
            cleaned_results, raw_results, validation_status = transform_outputs

            if cleaned_results:
                for db_id in cleaned_results:
                    try:
                        study_outdir = hash_outdir / db_id
                        study_outdir.mkdir(parents=True, exist_ok=True)

                        # Write study results using PipelineOutputsMixin method
                        self._write_study_results(
                            study_outdir,
                            db_id,
                            cleaned_results[db_id],
                            raw_results[db_id],
                        )

                        # Write study info including validation status
                        self._write_study_info(
                            hash_outdir=hash_outdir,
                            db_id=db_id,
                            study_inputs=all_study_inputs[db_id],
                            is_valid=validation_status[db_id],
                        )

                    except (IOError, OSError) as e:
                        msg = f"Failed to write results for study {db_id}: {str(e)}"
                        raise FileOperationError(msg)
                    except Exception as e:
                        raise ProcessingError(db_id, str(e))

        except (InputError, ProcessingError, ValidationError, FileOperationError) as e:
            logger.error(str(e))
            # Attempt cleanup on file operation failures
            if isinstance(e, FileOperationError):
                try:
                    import shutil

                    for study_id in all_study_inputs.keys():
                        study_dir = hash_outdir / study_id
                        if study_dir.exists():
                            shutil.rmtree(study_dir)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup after error: {str(cleanup_error)}")
            raise

    def _generate_config_hash_string(self, dataset: Dataset) -> str:
        """Create hash string including dataset keys for group processing.

        For dependent pipelines, we include dataset keys since all studies are
        processed together. This extends the base pipeline hash with dataset keys.
        """
        base_hash = super()._generate_config_hash_string(dataset)
        dataset_str = self.__serialize_dataset_keys(dataset)
        return f"{dataset_str}_{base_hash}"

    def __serialize_dataset_keys(self, dataset: Dataset) -> str:
        """Serialize dataset keys for hashing.

        For dependent pipelines, we need to include all study IDs in the hash
        since processing depends on the entire dataset.

        Args:
            dataset: Dataset object containing study data

        Returns:
            Comma-separated string of sorted study IDs
        """
        study_keys = sorted(dataset.data.keys())
        return "_".join(study_keys)


class IndependentPipeline(Pipeline):
    """Pipeline that processes each study independently."""

    def __process_and_write_study(
        self,
        study_data: Tuple[str, Dict[str, Any], Path],
        hash_outdir: Path,
        **kwargs,
    ) -> bool:
        """Process a single study and write its results."""
        db_id, study_inputs, study_outdir = study_data

        try:
            study_outdir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Load and process inputs
            try:
                loaded_inputs = {db_id: self._load_study_inputs(study_inputs)}
            except (IOError, ValueError) as e:
                raise InputError(f"Failed to load inputs for study {db_id}: {str(e)}")

            try:
                transform_results = self.transform(loaded_inputs, **kwargs)
                cleaned_results, raw_results, validation_status = transform_results
            except Exception as e:
                raise ProcessingError(db_id, str(e))

            if cleaned_results:
                try:
                    # Write study results using PipelineOutputsMixin method
                    self._write_study_results(
                        study_outdir, db_id, cleaned_results[db_id], raw_results[db_id]
                    )

                    # Write study info with validation status
                    self._write_study_info(
                        hash_outdir=hash_outdir,
                        db_id=db_id,
                        study_inputs=study_inputs,
                        is_valid=validation_status[db_id],
                    )
                except IOError as e:
                    msg = f"Failed to write results for study {db_id}: {str(e)}"
                    raise FileOperationError(msg)

            return True

        except (InputError, ProcessingError, ValidationError, FileOperationError) as e:
            logger.error(str(e))
            if isinstance(e, FileOperationError):
                try:
                    if study_outdir.exists():
                        import shutil

                        shutil.rmtree(study_outdir)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup after error: {str(cleanup_error)}")
            return False

    def _process_dataset(
        self,
        dataset: Dataset,
        hash_outdir: Path,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        """Process studies independently with optional parallelization."""
        # Filter and prepare studies that need processing
        filtered_dataset = self._filter_unprocessed_studies(hash_outdir, dataset)
        studies_to_process = []

        for db_id, study in filtered_dataset.data.items():
            study_inputs = self._collect_study_inputs(study)
            study_dir = hash_outdir / db_id

            studies_to_process.append((db_id, study_inputs, study_dir))

        if not studies_to_process:
            print("No studies need processing")
            return

        # Process and write results as they complete
        print(f"Processing {len(studies_to_process)} studies...")
        success_count = 0

        with tqdm(total=len(studies_to_process), desc="Processing studies") as pbar:
            if num_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as exc:
                    futures = []
                    for study_data in studies_to_process:
                        future = exc.submit(
                            self.__process_and_write_study,
                            study_data,
                            hash_outdir,
                            **kwargs,
                        )
                        future.add_done_callback(lambda _: pbar.update(1))
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        if future.result():
                            success_count += 1
            else:
                for study_data in studies_to_process:
                    if self.__process_and_write_study(
                        study_data, hash_outdir, **kwargs
                    ):
                        success_count += 1
                    pbar.update(1)

        print(
            f"Completed processing {success_count} "
            f"of {len(studies_to_process)} studies successfully"
        )


class Extractor(ABC):
    """Base class for data transformation logic.

    This class defines the core interface for transforming input study data into
    structured outputs validated against a schema. It separates the transformation
    logic from I/O operations (handled by Pipeline classes).

    Required Class Variables:
        _version: str - Version identifier for the extractor implementation
        _output_schema: Type[BaseModel] - Pydantic model defining expected output format

    Key Methods:
        _transform: Core transformation logic (must be implemented by subclasses)
        transform: Main entry point that coordinates transformation and validation
        validate_results: Validates outputs against the schema
        post_process: Optional hook for modifying results before validation

    Example:
        >>> class MyExtractor(Extractor[MyOutputSchema]):
        ...     _version = "1.0.0"
        ...     _output_schema = MyOutputSchema
        ...
        ...     def _transform(self, inputs: Dict[str, Dict[str, Any]], **kwargs):
        ...         # Transform input data
        ...         return transformed_data
    """

    _version: str = None
    _output_schema: Type[BaseModel] = None
    _normalize_fields: set[str] = set()
    _expand_abbrev_fields: set[str] = set()
    _nlp = None

    def __init__(
        self,
        nlp_model: str = "en_core_web_sm",
        disable_abbreviation_expansion: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize extractor and verify required configuration.

        Args:
            nlp_model: SpaCy model name for text processing. Defaults to "en_core_web_sm"
            disable_abbreviation_expansion:
                If True, disables abbreviation expansion
                even for fields with EXPAND_ABBREVIATIONS metadata.
                Defaults to False.
            **kwargs: Configuration parameters for the extractor

        Raises:
            ValueError: If _output_schema or _version not defined by subclass
        """
        self.disable_abbreviation_expansion = disable_abbreviation_expansion
        if not self._output_schema:
            raise ValueError("Subclass must define _output_schema class variable")
        if not self._version:
            raise ValueError("Subclass must define _version class variable")

        if isinstance(self, IndependentPipeline):
            IndependentPipeline.__init__(self, extractor=self)
        if isinstance(self, DependentPipeline):
            DependentPipeline.__init__(self, extractor=self)

        # Get fields needing processing
        # Get text processing fields
        fields = self._read_schema_metadata(self._output_schema)
        self._normalize_fields, self._expand_abbrev_fields = fields

        # Initialize NLP model if we have any fields needing abbreviation expansion
        if self._expand_abbrev_fields and not disable_abbreviation_expansion:
            import spacy

            try:
                self._nlp = spacy.load(nlp_model, disable=["parser", "ner"])
            except OSError:
                print(f"Downloading {nlp_model} model...")
                spacy.cli.download(nlp_model)
                self._nlp = spacy.load(nlp_model, disable=["parser", "ner"])

    def _read_schema_metadata(
        self, model: Type[BaseModel], prefix: str = ""
    ) -> Tuple[set[str], set[str]]:
        """Collect fields needing text normalization or abbreviation expansion.

        Recursively checks fields, tracking full path names (with dots):
        e.g., "groups.0.diagnosis" for a field in a list item

        Args:
            model: Pydantic model class to check
            prefix: Prefix for nested field names

        Returns:
            Tuple of:
            - set[str]: Field paths needing text normalization
            - set[str]: Field paths needing abbreviation expansion
        """
        normalize_fields = set()
        expand_abbrev_fields = set()

        for name, field in model.model_fields.items():
            # Build field path
            field_path = f"{prefix}.{name}" if prefix else name
            field_type = field.annotation

            # Check direct field metadata
            if field.json_schema_extra:
                if field.json_schema_extra.get(NORMALIZE_TEXT, False):
                    normalize_fields.add(field_path)
                if field.json_schema_extra.get(EXPAND_ABBREVIATIONS, False):
                    expand_abbrev_fields.add(field_path)

            # For iterable fields, append [] to path before recursing
            iterable_path = None
            nested_type = None

            # Handle List[Model]
            if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                nested_type = field_type.__args__[0]
                if hasattr(nested_type, "model_fields"):
                    iterable_path = f"{field_path}[]"

            # Handle Dict[str, Model]
            elif (
                hasattr(field_type, "__origin__")
                and field_type.__origin__ is dict
                and len(field_type.__args__) == 2
            ):
                nested_type = field_type.__args__[1]
                if hasattr(nested_type, "model_fields"):
                    iterable_path = f"{field_path}[]"

            # Handle regular nested model
            elif hasattr(field_type, "model_fields"):
                nested_type = field_type
                iterable_path = field_path

            # Recurse with proper path if we found a nested type
            if nested_type and iterable_path:
                nested_fields = self._read_schema_metadata(nested_type, iterable_path)
                normalize_fields.update(nested_fields[0])
                expand_abbrev_fields.update(nested_fields[1])

        return normalize_fields, expand_abbrev_fields

    @abstractmethod
    def _transform(
        self, inputs: Dict[str, Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Transform input data into output format.

        This is the core transformation method that must be implemented by subclasses.
        It should convert raw study data into the expected output format defined by T.

        Args:
            inputs: Dict mapping study IDs to their input data
                Format: {
                    "study_id": {
                        "input_type": input_data,
                        ...
                    }
                }
            **kwargs: Additional transformation arguments

        Returns:
            Dict mapping study IDs to their transformed outputs
            Format: {
                "study_id": transformed_data # type T
            }

        Raises:
            ProcessingError: If transformation fails for any study
        """
        pass

    def transform(
        self, study_inputs: Dict[str, Dict[str, Any]], **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, bool]]:
        """Transform and validate input data.

        This method orchestrates the complete transformation process:
        1. Calls _transform for core data processing
        2. Applies optional post-processing
        3. Validates results against schema

        Args:
            study_inputs: Dict mapping study IDs to their input data
                Format: {
                    "study_id": {
                        "input_type": input_data,
                        ...
                    }
                }
            **kwargs: Additional transformation arguments

        Returns:
            Tuple containing:
            - Dict[str, Dict[str, Any]]: Post-processed results keyed by study ID
            - Dict[str, Dict[str, Any]]: Raw results keyed by study ID
            - Validation status:
                - Dict[str, bool] for DependentPipeline (per-study validation)

        Raises:
            ProcessingError: If transformation fails
        """
        # Get raw results from transform
        raw_results = self._transform(study_inputs, **kwargs)

        # Ensure raw results maintain study ID structure
        if not isinstance(raw_results, dict):
            raise ProcessingError(
                None, "Transform must return dict with study IDs as keys"
            )

        # Post-process results with original inputs for abbreviation expansion
        cleaned_results = self.post_process(raw_results, study_inputs)

        # Validate each study's results individually
        validation_status = self.validate_results(cleaned_results)

        return (cleaned_results, raw_results, validation_status)

    def validate_results(self, results, **kwargs):
        """Apply validation to each study's results individually.

        Args:
            results: Dict mapping study IDs to their results
            **kwargs: Additional validation arguments

        Returns:
            Tuple of:
            - Dict mapping study IDs to their validated results
            - Dict mapping study IDs to validation status (True/False)
        """
        validation_status = {}

        for db_id, study_results in results.items():
            try:
                # Validate each study's results against the schema
                self._output_schema.model_validate(study_results)
                validation_status[db_id] = True
            except Exception as e:
                logging.error(f"Output validation error for study {db_id}: {str(e)}")
                validation_status[db_id] = False

        return validation_status

    def _process_field_value(
        self, value: str, normalize: bool, expand: bool, abbreviations: list
    ) -> str:
        """Process a single field value with the specified transformations."""
        if not isinstance(value, str):
            return value

        result = value

        if expand:
            result = resolve_abbreviations(result, abbreviations)
        if normalize:
            result = normalize_string(result)

        return result

    def _get_base_path_and_remainder(self, path: str) -> Tuple[str, str]:
        """Split a path at the first [] marker."""
        if "[]" in path:
            base, remainder = path.split("[]", 1)
            remainder = remainder.lstrip(".")
            return base, remainder
        return path, ""

    def post_process(
        self, results: Dict[str, Any], study_inputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Post-process transform results according to field metadata.

        Uses collected field paths to apply text processing to fields,
        handling iterable fields marked with [].

        Example paths:
            - groups[].diagnosis - Process diagnosis in each group
            - metadata.diagnosis - Process single diagnosis field

        Args:
            results: Dict mapping study IDs to their raw transformed data
            study_inputs: Dict containing input data with source text

        Returns:
            Dict with same structure as input, with processed text fields
        """
        processed_results = {}

        for study_id, study_data in results.items():
            # Deep copy the data to avoid modifying original
            processed_study = json.loads(json.dumps(study_data))

            # Extract abbreviations once per study if needed
            study_abbreviations = []
            if (
                not self.disable_abbreviation_expansion
                and self._expand_abbrev_fields
                and study_id in study_inputs
            ):
                if "text" in study_inputs[study_id]:
                    source_text = study_inputs[study_id].get("text", None)
                    if isinstance(source_text, str) and self._nlp is not None:
                        study_abbreviations = load_abbreviations(
                            source_text, model=self._nlp
                        )

            # Find which transformations each path needs
            field_transforms = {
                path: (
                    path in self._normalize_fields,
                    path in self._expand_abbrev_fields
                    and not self.disable_abbreviation_expansion,
                )
                for path in self._normalize_fields.union(self._expand_abbrev_fields)
            }

            # Process each path
            for path, (do_normalize, do_expand) in field_transforms.items():
                # Split path at first [] if present
                base_path, remainder = self._get_base_path_and_remainder(path)

                # Get base value
                base_value = reduce(
                    lambda d, k: d.get(k, {}) if isinstance(d, dict) else d,
                    base_path.split("."),
                    processed_study,
                )

                # Handle iterables
                if remainder:
                    if isinstance(base_value, (list, dict)):
                        # Recursively process each item
                        items = (
                            base_value.values()
                            if isinstance(base_value, dict)
                            else base_value
                        )
                        for item in items:
                            # Process and update nested field
                            current = item
                            parts = remainder.split(".")
                            for i, part in enumerate(parts):
                                # Last part is the field to process
                                if i == len(parts) - 1:
                                    if isinstance(current.get(part), str):
                                        current[part] = self._process_field_value(
                                            current[part],
                                            do_normalize,
                                            do_expand,
                                            study_abbreviations,
                                        )
                                # Navigate through intermediate parts
                                else:
                                    current = current.get(part, {})
                else:
                    # Direct field processing
                    if isinstance(base_value, str):
                        new_value = self._process_field_value(
                            base_value, do_normalize, do_expand, study_abbreviations
                        )
                        # Set the processed value
                        current = processed_study
                        parts = base_path.split(".")
                        for part in parts[:-1]:
                            current = current[part]
                        current[parts[-1]] = new_value

            processed_results[study_id] = processed_study

        return processed_results
