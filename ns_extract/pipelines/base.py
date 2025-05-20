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
from ns_extract.pipelines.normalize import (
    normalize_string,
    load_abbreviations,
    resolve_abbreviations,
)
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
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel
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
from ns_extract.pipelines.data_structures import (
    InputPipelineInfo,
    PipelineOutputInfo,
    StudyOutputJson,
)


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
        study_inputs = self.gather_all_study_inputs(dataset)

        for db_id, study in dataset.data.items():
            # Get existing input file hashes for this study
            existing = existing_results.get(db_id, {}).get("inputs", {})

            # Skip if no existing results or no current inputs
            if not existing or db_id not in study_inputs:
                result_matches[db_id] = False
                continue

            # Use _are_file_hashes_identical to compare hashes
            result_matches[db_id] = self._are_file_hashes_identical(
                study_inputs[db_id], existing
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
        input_pipeline_info: Optional[Dict[str, Dict[str, str]]] = None,
        num_workers=1,
        **kwargs,
    ):
        """Process individual studies through the pipeline independently."""
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
                input_pipelines=input_pipeline_info or {},
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

    def _serialize_dataset_keys(self, dataset: Dataset) -> str:
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

    def check_for_changes(
        self,
        hash_outdir: Path,
        dataset: Dataset,
        input_pipeline_info: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self._filter_existing_results(hash_outdir, dataset)
        matching_results = self._identify_matching_results(
            dataset,
            existing_results,
        )
        # Return True if any of the studies' inputs have changed or if new studies exist
        return any(not match for match in matching_results.values())

    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[Dict[str, Dict[str, str]]] = None,
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
            input_pipelines=input_pipeline_info or {},
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

            # Process loaded inputs with fit_transform
            transform_outputs = self.fit_transform(loaded_study_inputs, **kwargs)
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

    # Removed validate_results method - now using unified validate() method


class Extractor(ABC):
    """Base class for data transformation logic.

    This class defines the core interface for transforming input study data into
    structured outputs validated against a schema. It follows the scikit-learn
    fit/transform pattern while supporting both pre-trained and trainable models.

    Required Class Variables:
        _version: str - Version identifier for the extractor implementation
        _output_schema: Type[BaseModel] - Pydantic model defining expected output format

    Key Methods:
        fit: Train the extractor on input data (must be implemented by subclasses)
        transform: Transform input data (must be implemented by subclasses)
        fit_transform: Convenience method to fit and transform in one step
        validate: Validate outputs against the schema
        post_process: Optional hook for modifying results before validation

    The interface handles both independent and dependent pipeline processing:
    - Independent: processes individual studies (single dict input)
    - Dependent: processes all studies together (dict of dicts input)

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

    def __init__(
            self,
            expand_abbreviations_fields: List = [],
            normalizable_string_fields: List = [],
            **kwargs: Any) -> None:
        """Initialize extractor and verify required configuration.

        Args:
            expand_abbreviations_fields: List of fields to apply normalization to
            **kwargs: Configuration parameters for the extractor

        Raises:
            ValueError: If _output_schema or _version not defined by subclass
        """
        if not self._output_schema:
            raise ValueError("Subclass must define _output_schema class variable")
        if not self._version:
            raise ValueError("Subclass must define _version class variable")

        self.normalizable_string_fields = normalizable_string_fields
        self.expand_abbreviations_fields = expand_abbreviations_fields
        self._nlp = None

        # Pre-load NLP model if we'll need it
        if expand_abbreviations_fields:
            import spacy
            try:
                self._nlp = spacy.load("en_core_sci_sm", disable=["parser", "ner"])
                if "abbreviation_detector" not in self._nlp.pipe_names:
                    import scispacy  # noqa: F401
                    self._nlp.add_pipe("abbreviation_detector")
            except Exception as e:
                print(f"Warning: Failed to load NLP model: {e}")

        if isinstance(self, IndependentPipeline):
            IndependentPipeline.__init__(self, extractor=self)
        if isinstance(self, DependentPipeline):
            DependentPipeline.__init__(self, extractor=self)

    @abstractmethod
    def _transform(
        self, inputs: Dict[str, Dict[str, Any]], **kwargs: Any
    ) -> None:
        """Implementation of model training logic.

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
                For independent pipelines, will contain single study.
                For dependent pipelines, will contain all studies.
            **kwargs: Additional training arguments

        Raises:
            ProcessingError: If training fails
        """
        pass

    def _process_text_field(
        self,
        text: str,
        field_name: str,
        abbreviations: Optional[List[Dict]] = None
    ) -> str:
        """Process a text field with abbreviation expansion and/or normalization."""
        if not isinstance(text, str):
            return text

        # First expand abbreviations if needed
        if field_name in self.expand_abbreviations_fields and abbreviations:
            text = resolve_abbreviations(text, abbreviations)

        # Then normalize if needed
        if field_name in self.normalizable_string_fields:
            text = normalize_string(text)
            
        return text

    def _normalize_nested_fields(
        self,
        data: Any,
        abbreviations: Optional[List[Dict]] = None
    ) -> Any:
        """Recursively process fields in nested data structures."""
        if isinstance(data, dict):
            normalized = {}
            for key, value in data.items():
                if isinstance(value, str):
                    normalized[key] = self._process_text_field(value, key, abbreviations)
                else:
                    normalized[key] = self._normalize_nested_fields(value, abbreviations)
            return normalized
        elif isinstance(data, list):
            return [self._normalize_nested_fields(item, abbreviations) for item in data]
        return data

    def post_process(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Optional hook for post-processing transform results.

        This can be overridden by subclasses to modify results before validation,
        for example to clean or normalize data.

        Args:
            results: Dict mapping study IDs to their raw transformed data

        Returns:
            Dict with same structure as input, potentially modified
        """
        if isinstance(self, DependentPipeline):
            # Process each study with its own abbreviations
            processed_results = {}
            for study_id, study_data in results.items():
                # Extract abbreviations from this study's source text
                source_inputs = next(s for s in study_data.values() if isinstance(s, dict))
                abbreviations = self._extract_source_abbreviations(source_inputs)
                # Process this study's data with its abbreviations
                processed_results[study_id] = self._normalize_nested_fields(
                    study_data, abbreviations
                )
            return processed_results
        else:
            # Process single study with its abbreviations
            abbreviations = self._extract_source_abbreviations(results)
            return self._normalize_nested_fields(results, abbreviations)

    def validate(self, output: Dict[str, Any]) -> Union[bool, Dict[str, bool]]:
        """Validate transformed data against schema.

        Unified validation method that handles both independent and dependent
        pipeline cases.

        Args:
            output: Dict mapping study IDs to transformed data

        Returns:
            For independent pipelines: bool indicating if validation passed
            For dependent pipelines: Dict mapping study IDs to validation status

        Note:
            Validation failures are logged but don't raise exceptions
            to allow graceful handling of invalid results
        """
        if isinstance(self, DependentPipeline):
            # Validate each study individually
            validation_status = {}
            for study_id, study_output in output.items():
                try:
                    self._output_schema.model_validate(study_output)
                    validation_status[study_id] = True
                except Exception as e:
                    logging.error(f"Output validation error for study {study_id}: {str(e)}")
                    validation_status[study_id] = False
            return validation_status
        else:
            # Validate single output
            try:
                self._output_schema.model_validate(output)
                return True
            except Exception as e:
                logging.error(f"Output validation error: {str(e)}")
                return False

    def _extract_source_abbreviations(self, inputs: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """Extract abbreviations from the source text (ace or pubget).

        Args:
            inputs: Dict containing input data with source text

        Returns:
            List of abbreviation dictionaries from the source text
        """
        if not self._nlp or not self.expand_abbreviations_fields:
            return []

        # Look for text in data pond inputs with priority order
        for sources, input_types in self._data_pond_inputs.items():
            if "text" not in input_types:
                continue
                
            for source in sources:  # e.g., try pubget first, then ace
                if source not in inputs:
                    continue
                    
                if "text" not in inputs[source]:
                    continue
                    
                source_text = inputs[source]["text"]
                if not isinstance(source_text, str):
                    continue
                    
                # Found valid source text, extract abbreviations
                return load_abbreviations(source_text, model=self._nlp)
                
        return []
