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
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from ns_extract.dataset import Dataset
from ns_extract.pipelines.utils import (
    FileOperationsMixin,
    StudyInputsMixin,
    PipelineOutputsMixin,
)
from ns_extract.pipelines.exceptions import (
    FileOperationError,
    InputError,
    ProcessingError,
    ValidationError,
)

# Type variables for generics
T = TypeVar("T")  # Generic type for pipeline data
S = TypeVar("S", bound="BaseModel")  # Schema type variable bounded to BaseModel

# Type aliases with documentation
PipelineConfig = Dict[str, Dict[str, str]]  # Pipeline configuration mapping
StudyResults = Dict[str, Any]  # Results for a single study
PipelineResults = Dict[str, StudyResults]  # Results across multiple studies
ValidationResult = Tuple[bool, StudyResults]  # Validation result with data
PipelineInputs = Dict[Tuple[str, ...], Tuple[str, ...]]  # Input source -> type mapping

# Constants
PIPELINE_VERSION = "latest"  # Default pipeline version
CONFIG_VERSION = "latest"  # Default config version

# Input type constants
SOURCE_INPUTS = ["text", "coordinates", "metadata"]
PIPELINE_INPUTS = ["results", "raw_results", "info"]


logger = logging.getLogger(__name__)


class InputPipelineInfo(BaseModel):
    """Information about the input pipeline."""

    pipeline_dir: Path = Field(description="Path to the pipeline directory")
    version: str = Field(description="Version of the pipeline")
    config_hash: str = Field(description="Hash of the pipeline configuration")


class PipelineOutputInfo(BaseModel):
    """Information about the pipeline output."""

    date: str = Field(description="Date of the output")
    version: str = Field(description="Version of the pipeline")
    config_hash: str = Field(description="Hash of the pipeline configuration")
    extractor: str = Field(description="Name of the extractor used")
    extractor_kwargs: Dict[str, Any] = Field(
        description="Arguments passed to the extractor"
    )
    transform_kwargs: Dict[str, Any] = Field(
        description="Arguments passed to the transform function"
    )
    input_pipelines: Dict[str, InputPipelineInfo] = Field(
        description="Pipelines used as inputs to this pipeline"
    )
    schema: Dict[str, Any] = Field(description="Schema of the output data")


class StudyOutputJson(BaseModel):
    """Information about a study's processing results."""

    date: str = Field(description="When the study was processed")
    inputs: Dict[str, str] = Field(description="Input file paths and their MD5 hashes")
    valid: bool = Field(description="Whether outputs passed validation")


def deep_getattr(obj: Any, attr_path: str, default: Any = None) -> Any:
    try:
        return reduce(getattr, attr_path.split("."), obj)
    except AttributeError:
        return default


class Pipeline(FileOperationsMixin, StudyInputsMixin, PipelineOutputsMixin):
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

    _data_pond_inputs: PipelineInputs = Field(default_factory=dict)
    _input_pipelines: PipelineInputs = Field(default_factory=dict)

    def __init__(
        self,
        extractor: Extractor,
    ):
        """Initialize pipeline.

        Args:
            extractor: Extractor instance that defines the data transformation
        """
        self.extractor = extractor

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
        args = list(inspect.signature(self.__init__).parameters.keys())
        args_str = "_".join([f"{arg}_{str(getattr(self, arg))}" for arg in args])
        version_str = self.extractor._version

        return f"{dataset_str}_{version_str}_{args_str}"

    def _prepare_pipeline(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[InputPipelineInfo] = None,
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
            self._create_hash_string(dataset).encode()
        ).hexdigest(6)

        # Create full directory path
        hash_outdir = (
            Path(output_directory)
            / self.extractor.__class__.__name__
            / self.extractor._version
            / hash_str
        )

        return hash_outdir

    @abstractmethod
    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[Dict[str, InputPipelineInfo]] = None,
        **kwargs: Any,
    ) -> None:
        """Process a dataset through the pipeline.

        Args:
            dataset: Dataset to process
            output_directory: Directory to write outputs
            input_pipeline_info: Optional configuration for input pipelines. Maps pipeline
                names to their configuration info.
            **kwargs: Additional arguments passed to extractor for data transformation

        Raises:
            FileOperationError: If directory creation or file operations fail
            ProcessingError: If pipeline processing fails
            ValidationError: If results fail schema validation
        """
        pass

    def _filter_existing_results(
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

            pipeline_info = self.load_json(d / "pipeline_info.json")
            pipeline_args = pipeline_info.get("extractor_kwargs", {})

            if pipeline_args != current_args:
                continue

            for sub_d in d.glob("*"):
                if not sub_d.is_dir():
                    continue

                info_file = sub_d / "info.json"
                if info_file.exists():
                    info = self.load_json(info_file)
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

    def _are_file_hashes_identical(
        self, study_inputs: Dict[str, Path], existing_inputs: Dict[str, str]
    ) -> bool:
        """Compare file hashes to determine if the inputs have changed."""
        if set(str(p) for p in study_inputs.values()) != set(existing_inputs.keys()):
            return False

        for existing_file, hash_val in existing_inputs.items():
            if self.calculate_md5(Path(existing_file)) != hash_val:
                return False

        return True

    def gather_all_study_inputs(self, dataset: Dataset) -> Dict[str, Dict[str, Path]]:
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
            db_id: self.collect_study_inputs(study)
            for db_id, study in dataset.data.items()
        }

    def _identify_matching_results(
        self,
        dataset: Dataset,
        existing_results: Dict[str, Dict[str, Dict[str, str]]],
        input_pipeline_info: Optional[dict] = None,
    ) -> Dict[str, bool]:
        """Compare dataset inputs with existing results to identify changes.

        Args:
            dataset: Dataset to check for changes
            existing_results: Previously processed results to compare against
                Format matches return type of _filter_existing_results
            input_pipeline_info: Optional information about input pipelines

        Returns:
            Dict mapping study IDs to boolean indicating if inputs match
            existing results (True = no changes, False = changes detected)
        """
        result_matches = {}
        for db_id, study in dataset.data.items():
            # Get existing input file hashes for this study
            existing = existing_results.get(db_id, {}).get("inputs", {})

            # Collect all input files and their current hashes
            current_inputs = {}

            # Add data pond input files and their hashes
            study_inputs = self.gather_all_study_inputs(dataset)
            for input_path in study_inputs.get(db_id, {}).values():
                current_inputs[str(input_path)] = self.calculate_md5(input_path)

            # Add pipeline input files and their hashes
            if input_pipeline_info:
                for pipeline_name in input_pipeline_info:
                    if pipeline_name not in study.pipeline_results:
                        result_matches[db_id] = False
                        break
                    result_path = study.pipeline_results[pipeline_name].result
                    current_inputs[str(result_path)] = self.calculate_md5(result_path)

            # Compare all input hashes
            if db_id not in result_matches:  # Skip if already marked as not matching
                result_matches[db_id] = set(current_inputs.items()) == set(
                    existing.items()
                )

        return result_matches

    def filter_inputs(self, hash_outdir: Path, dataset: Dataset) -> Dataset:
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
        existing_results = self._filter_existing_results(hash_outdir, dataset)
        matching_results = self._identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        keep_ids = set(dataset.data.keys()) - {
            db_id for db_id, match in matching_results.items() if match
        }
        return dataset.slice(keep_ids)

    def collect_study_inputs(
        self, study: Any, input_pipeline_info: Optional[InputPipelineInfo] = None
    ) -> Dict[str, Path]:
        """Collect inputs for a study."""
        study_inputs = {}

        # Iterate through sources in priority order
        for sources, input_types in self._data_pond_inputs.items():
            for source in sources:
                source_obj = getattr(study, source, None)
                if source_obj:
                    for input_type in input_types:
                        input_obj = deep_getattr(source_obj, input_type, None)
                        if input_obj and input_type not in study_inputs:
                            study_inputs[input_type] = input_obj
                    break  # Stop after finding first valid source

        # Add pipeline inputs if needed
        if input_pipeline_info:
            for name in input_pipeline_info:
                result_path = study.pipeline_results[name].result
                study_inputs[name] = result_path

        return study_inputs

    def write_study_info(
        self,
        hash_outdir: Path,
        db_id: str,
        study_inputs: Dict[str, Path],
        is_valid: bool,
    ):
        """Write information about the study run to info.json file."""
        info = StudyOutputJson(
            date=datetime.now().isoformat(),
            inputs={
                str(input_file): self.calculate_md5(input_file)
                for input_file in study_inputs.values()
            },
            valid=is_valid,
        )
        self.write_json(hash_outdir / db_id / "info.json", info.model_dump())

    def _write_results(self, hash_outdir: Path, results: Dict[str, Any]) -> None:
        """Write pipeline results and metadata.

        Handles writing both raw and processed results when available.
        If extractor's post_process modifies results, both versions are written:
        - raw_results.json: Original transform output
        - results.json: Post-processed output

        Args:
            hash_outdir: Directory to write results to
            results: Results from extractor to write

        Raises:
            IOError: If writing any file fails
            ValueError: If hash_outdir is not a Path
        """
        if not isinstance(hash_outdir, Path):
            raise ValueError("hash_outdir must be a Path object")

        # Write each study's results
        for study_id, study_results in results.items():
            study_dir = hash_outdir / study_id
            study_dir.mkdir(exist_ok=True, parents=True)
            self.write_study_results(study_dir, study_id, study_results)


class IndependentPipeline(Pipeline):
    """Pipeline that processes each study independently."""

    def _process_and_write_study(
        self,
        study_data: Tuple[str, Dict[str, Any], Path],
        hash_outdir: Path,
        **kwargs,
    ) -> bool:
        """Process a single study and write its results.

        Args:
            study_data: Tuple of (db_id, study_inputs, study_outdir)
            hash_outdir: Base output directory
            **kwargs: Additional arguments for processing

        Returns:
            bool: True if processing was successful

        Raises:
            ProcessingError: If study processing fails
            FileOperationError: If writing results fails
            ValidationError: If results fail validation
        """
        db_id, study_inputs, study_outdir = study_data

        try:
            study_outdir.mkdir(parents=True, exist_ok=True)

            # Process inputs with StudyInputsManager
            try:
                loaded_study_inputs = self.load_study_inputs(
                    study_inputs=study_inputs,
                )
            except (IOError, ValueError) as e:
                raise InputError(f"Failed to load inputs for study {db_id}: {str(e)}")

            # Process the inputs
            try:
                results, raw_results, is_valid = self.transform(
                    loaded_study_inputs, **kwargs
                )
            except Exception as e:
                raise ProcessingError(db_id, str(e))

            if not is_valid:
                raise ValidationError(f"Results validation failed for study {db_id}")

            if results:
                # Write results immediately
                try:
                    self.write_study_info(
                        hash_outdir=hash_outdir,
                        db_id=db_id,
                        study_inputs=study_inputs,
                        is_valid=is_valid,
                    )
                    self.write_json(study_outdir / "results.json", results)

                    if raw_results is not results:
                        self.write_json(study_outdir / "raw_results.json", raw_results)
                except IOError as e:
                    msg = f"Failed to write results for study {db_id}: {str(e)}"
                    raise FileOperationError(msg)

            return True

        except (InputError, ProcessingError, ValidationError, FileOperationError) as e:
            logger.error(str(e))
            if isinstance(e, FileOperationError):
                # Attempt cleanup on file operation failures
                try:
                    import shutil

                    if study_outdir.exists():
                        shutil.rmtree(study_outdir)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup after error: {str(cleanup_error)}")
            return False

    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[InputPipelineInfo] = None,
        num_workers=1,
        **kwargs,
    ):
        """Process individual studies through the pipeline independently."""
        # TODO: make sure input_pipeline_info are handled with the hashing
        # should either need to find the latest hash, or just use the existing
        # name/version/hash. There could be additional studies that are added
        # but for independent pipelines, we don't need to worry about that,
        # it will come into play when we need to filter the inputs for which studies
        # need to be processed. With dependent pipelines, if there were new studies,
        # then there would be a new hash created with all the study results.
        hash_outdir = self._prepare_pipeline(
            dataset, output_directory, input_pipeline_info
        )

        if not (hash_outdir / "pipeline_info.json").exists():
            hash_outdir.mkdir(parents=True, exist_ok=True)
            # Create pipeline info object
            pipeline_info = PipelineOutputInfo(
                date=datetime.now().isoformat(),
                version=self.extractor._version,
                config_hash=hash_outdir.name,
                extractor=self.extractor.__class__.__name__,
                extractor_kwargs={
                    arg: getattr(self, arg)
                    for arg in inspect.signature(self.__init__).parameters.keys()
                },
                transform_kwargs=kwargs,
                input_pipelines=self.convert_pipeline_info(input_pipeline_info),
                schema=self.extractor._output_schema.model_json_schema(),
            )
            self.write_pipeline_info(hash_outdir, pipeline_info)

        # Filter and prepare studies that need processing
        filtered_dataset = self.filter_inputs(hash_outdir, dataset)
        studies_to_process = []

        for db_id, study in filtered_dataset.data.items():
            study_inputs = self.collect_study_inputs(study, input_pipeline_info)
            # Create study directory directly under hash directory
            study_dir = hash_outdir / db_id

            # Skip if already processed
            if study_dir.exists():
                continue

            studies_to_process.append((db_id, study_inputs, study_dir))

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
        hash_outdir: Path,
        dataset: Dataset,
        input_pipeline_info: Optional[InputPipelineInfo] = None,
    ) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self._filter_existing_results(hash_outdir, dataset)
        matching_results = self._identify_matching_results(
            dataset, existing_results, input_pipeline_info=input_pipeline_info
        )
        # Return True if any of the studies' inputs have changed or if new studies exist
        return any(not match for match in matching_results.values())

    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[InputPipelineInfo] = None,
        **kwargs,
    ):
        """Process all studies through the pipeline as a group."""
        # Get initial hash directory
        hash_outdir = self._prepare_pipeline(
            dataset, output_directory, input_pipeline_info
        )

        # Create pipeline info for checking changes
        pipeline_info = PipelineOutputInfo(
            date=datetime.now().isoformat(),
            version=self.extractor._version,
            config_hash=hash_outdir.name,
            extractor=self.extractor.__class__.__name__,
            extractor_kwargs={
                arg: getattr(self, arg)
                for arg in inspect.signature(self.__init__).parameters.keys()
            },
            transform_kwargs=kwargs,
            input_pipelines=self.convert_pipeline_info(input_pipeline_info),
            schema=self.extractor._output_schema.model_json_schema(),
        )

        # If directory exists and has no changes, we're done
        if hash_outdir.exists():
            if not (hash_outdir / "pipeline_info.json").exists():
                self.write_pipeline_info(hash_outdir, pipeline_info)
            if not self.check_for_changes(hash_outdir, dataset, input_pipeline_info):
                logger.info("No changes detected, skipping pipeline execution.")
                return
            # Changes detected, create new directory
            hash_outdir = self.get_next_available_dir(hash_outdir)
            hash_outdir.mkdir(parents=True)
            self.write_pipeline_info(hash_outdir, pipeline_info)
        else:
            # First run, create directory and write info
            hash_outdir.mkdir(parents=True)
            self.write_pipeline_info(hash_outdir, pipeline_info)

        # Collect all inputs and run the group function at once
        try:
            all_study_inputs = {
                db_id: self.collect_study_inputs(study, input_pipeline_info)
                for db_id, study in dataset.data.items()
            }

            # Load all study inputs
            loaded_study_inputs = {}
            for db_id, study_inputs in all_study_inputs.items():
                try:
                    loaded_study_inputs[db_id] = self.load_study_inputs(study_inputs)
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

                        # Write cleaned results
                        self.write_json(
                            study_outdir / "results.json", cleaned_results[db_id]
                        )

                        # Write raw results if different
                        if raw_results[db_id] != cleaned_results[db_id]:
                            self.write_json(
                                study_outdir / "raw_results.json", raw_results[db_id]
                            )

                        # Write study info including validation status
                        self.write_study_info(
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
        validated_results = {}
        validation_status = {}

        for db_id, study_results in results.items():
            try:
                # Validate each study's results against the schema
                self._output_schema.model_validate(study_results)
                validated_results[db_id] = study_results
                validation_status[db_id] = True
            except Exception as e:
                logging.error(f"Output validation error for study {db_id}: {str(e)}")
                validated_results[db_id] = study_results
                validation_status[db_id] = False

        return validated_results, validation_status


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
        validate_output: Validates outputs against the schema
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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize extractor and verify required configuration.

        Args:
            **kwargs: Configuration parameters for the extractor

        Raises:
            ValueError: If _output_schema or _version not defined by subclass
        """
        if not self._output_schema:
            raise ValueError("Subclass must define _output_schema class variable")
        if not self._version:
            raise ValueError("Subclass must define _version class variable")

        if isinstance(self, IndependentPipeline):
            IndependentPipeline.__init__(self, extractor=self)
        if isinstance(self, DependentPipeline):
            DependentPipeline.__init__(self, extractor=self)

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
        self, inputs: Dict[str, Dict[str, Any]], **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Union[bool, Dict[str, bool]]]:
        """Transform and validate input data.

        This method orchestrates the complete transformation process:
        1. Calls _transform for core data processing
        2. Applies optional post-processing
        3. Validates results against schema

        Args:
            inputs: Dict mapping study IDs to their input data
            **kwargs: Additional transformation arguments

        Returns:
            Tuple containing:
            - Post-processed results (Dict[str, T])
            - Raw results before post-processing (Dict[str, T])
            - Validation status:
                - bool for IndependentPipeline (single result)
                - Dict[str, bool] for DependentPipeline (per-study validation)

        Raises:
            ProcessingError: If transformation fails
        """
        # Get raw results from transform
        raw_results = self._transform(inputs, **kwargs)

        # Post-process results
        cleaned_results = self.post_process(raw_results)

        # Validate results based on pipeline type
        if isinstance(self, DependentPipeline):
            # For dependent pipelines, validate each study individually
            _, validation_status = self.validate_results(cleaned_results)
        else:
            # For independent pipelines, validate single study
            validation_status = self.validate_output(cleaned_results)

        return (cleaned_results, raw_results, validation_status)

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate transformed data against schema.

        Args:
            output: Dict mapping study IDs to transformed data

        Returns:
            bool: True if validation passes, False otherwise

        Note:
            Validation failures are logged but don't raise exceptions
            to allow graceful handling of invalid results
        """
        try:
            self._output_schema.model_validate(output)
            return True
        except Exception as e:
            logging.error(f"Output validation error: {str(e)}")
            return False

    def post_process(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Optional hook for post-processing transform results.

        This can be overridden by subclasses to modify results before validation,
        for example to clean or normalize data.

        Args:
            results: Dict mapping study IDs to their raw transformed data

        Returns:
            Dict with same structure as input, potentially modified
        """
        return results
