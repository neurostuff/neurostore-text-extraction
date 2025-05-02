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
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from ns_extract.dataset import Dataset, Study

# Type aliases
PipelineConfig = Dict[str, Dict[str, str]]  # Pipeline configuration
StudyResults = Dict[str, Any]  # Single study results
PipelineResults = Dict[str, StudyResults]  # Multiple study results
ValidationResult = Tuple[bool, StudyResults]  # Validation result format
PipelineInputs = Dict[Tuple[str], Tuple[str]]  # Pipeline input types

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


class StudyInputsManager:
    """Static methods for handling study input file operations."""

    @staticmethod
    def load_text_file(file_path: Path) -> str:
        """Read text file with error handling."""
        try:
            with file_path.open("r") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read text file {file_path}: {str(e)}")

    @staticmethod
    def load_study_inputs(study_inputs: Dict[str, Union[str, Path]]) -> Dict[str, Any]:
        """Load study input files based on their file extensions.

        Args:
            study_inputs: Dictionary mapping input names to file paths

        Returns:
            Dictionary with loaded file contents

        Raises:
            IOError: If file reading fails
            ValueError: If file extension not supported
        """
        loaded_inputs = {}
        for input_name, file_path in study_inputs.items():
            path = Path(file_path)
            suffix = path.suffix.lower()

            try:
                if suffix == ".txt":
                    loaded_inputs[input_name] = StudyInputsManager.load_text_file(path)
                elif suffix == ".json":
                    loaded_inputs[input_name] = FileManager.load_json(path)
                elif suffix == ".csv":
                    import pandas as pd

                    loaded_inputs[input_name] = pd.read_csv(path).to_dict("records")
                else:
                    raise ValueError(
                        f"Unsupported file type for {input_name}: {suffix}"
                    )
            except Exception as e:
                raise IOError(
                    f"Failed to load study input {input_name} from {path}: {str(e)}"
                )

        return loaded_inputs

    def collect_dataset_inputs(
        self, dataset: Dataset, input_pipeline_info: Optional[PipelineInputs] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Collect all required inputs for dataset processing.

        This method coordinates input collection across all studies,
        gathering both source file inputs and pipeline dependencies.

        Args:
            dataset: Dataset to collect inputs for
            input_pipeline_info: Optional pipeline configuration arguments
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
            study_id: self._collect_study_inputs(study, input_pipeline_info)
            for study_id, study in dataset.data.items()
        }

    def _collect_study_inputs(
        self, study: Study, input_pipeline_info: Optional[PipelineConfig] = None
    ) -> Dict[str, Any]:
        """Collect all required inputs for a single study.

        Args:
            study: Study to collect inputs for
            input_pipeline_info: Optional pipeline configuration arguments

        Returns:
            Dict of collected study inputs
        """
        inputs = {}

        # Get source file inputs
        for source_type in SOURCE_INPUTS:
            if hasattr(study, source_type):
                inputs[source_type] = getattr(study, source_type)

        # Get pipeline dependency inputs
        if input_pipeline_info:
            for name in input_pipeline_info:
                result_path = study.pipeline_results[name].result
                inputs[name] = result_path

        return inputs


class PipelineOutputsManager:
    """Static methods for managing pipeline output files."""

    @staticmethod
    def convert_pipeline_info(
        info: Dict[str, Dict[str, str]]
    ) -> Dict[str, InputPipelineInfo]:
        """Convert pipeline info dict to InputPipelineInfo objects."""
        if info is None:
            return {}

        return {name: InputPipelineInfo(**kwargs) for name, kwargs in info.items()}

    @staticmethod
    def create_pipeline_info(
        extractor_name: str,
        extractor_kwargs: Dict[str, Any],
        version: str,
        config_hash: str,
        output_schema: Type[BaseModel],
        input_pipeline_info: Optional[Dict[str, Dict[str, str]]] = None,
        transform_kwargs: Dict[str, Any] = None,
    ) -> PipelineOutputInfo:
        """Create PipelineOutputInfo instance with provided data.

        Args:
            extractor_name: Name of the extractor
            extractor_kwargs: Keyword arguments used to initialize extractor
            version: Pipeline version
            config_hash: Hash of the pipeline configuration
            output_schema: Schema for output validation
            input_pipeline_info: Optional dict of input pipeline configurations
            transform_kwargs: Optional transform function arguments

        Returns:
            PipelineOutputInfo instance with normalized data
        """
        return PipelineOutputInfo(
            date=datetime.now().isoformat(),
            version=version,
            config_hash=config_hash,
            extractor=extractor_name,
            extractor_kwargs=extractor_kwargs or {},
            transform_kwargs=transform_kwargs or {},
            input_pipelines=PipelineOutputsManager.convert_pipeline_info(
                input_pipeline_info
            ),
            schema=output_schema.model_json_schema(),
        )

    @staticmethod
    def validate_results(results: Dict[str, Any]) -> bool:
        """Validate that results have the expected structure."""
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

    @staticmethod
    def write_pipeline_info(
        hash_outdir: Path,
        info: PipelineOutputInfo,
    ) -> None:
        """Write pipeline metadata to pipeline_info.json."""
        if not isinstance(hash_outdir, Path):
            raise ValueError("output_dir must be a Path object")

        try:
            info_path = hash_outdir / "pipeline_info.json"
            FileManager.write_json(info_path, info.model_dump())
        except IOError as e:
            logger.error(f"Failed to write pipeline info: {str(e)}")
            raise

    @staticmethod
    def write_study_results(
        study_dir: Path, study_id: str, results: Dict[str, Any]
    ) -> None:
        """Write study results to directory."""
        if not isinstance(study_dir, Path):
            raise ValueError("study_dir must be a Path object")
        if not study_id:
            raise ValueError("study_id cannot be empty")
        if not PipelineOutputsManager.validate_results(results):
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


class Pipeline:
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

        # Create directory if needed
        hash_outdir.mkdir(parents=True, exist_ok=True)

        return hash_outdir

    @abstractmethod
    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_info: Optional[InputPipelineInfo] = None,
        **kwargs,
    ):
        """Process a dataset through the pipeline.

        Args:
            dataset: Dataset to process
            output_directory: Directory to write outputs
            input_pipeline_info: Optional kwargs for input pipelines
            **kwargs: Additional arguments passed to extractor
        """
        pass

    def _filter_existing_results(
        self, hash_outdir: Path, dataset: Any
    ) -> Dict[str, Dict]:
        """Find the most recent result for an existing study."""
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

            pipeline_info = FileManager.load_json(d / "pipeline_info.json")
            pipeline_args = pipeline_info.get("extractor_kwargs", {})

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
            if FileManager.calculate_md5(Path(existing_file)) != hash_val:
                return False

        return True

    def gather_all_study_inputs(self, dataset: Any) -> Dict[str, Dict[str, Path]]:
        """Collect all inputs for the dataset."""
        return {
            db_id: self.collect_study_inputs(study)
            for db_id, study in dataset.data.items()
        }

    def _identify_matching_results(
        self, dataset: Any, existing_results: Dict[str, Dict]
    ) -> Dict[str, bool]:
        """Compare dataset inputs with existing results."""
        dataset_inputs = self.gather_all_study_inputs(dataset)
        return {
            db_id: self._are_file_hashes_identical(
                study_inputs, existing_results.get(db_id, {}).get("inputs", {})
            )
            for db_id, study_inputs in dataset_inputs.items()
        }

    def filter_inputs(self, hash_outdir: Path, dataset: Any) -> bool:
        """Filter inputs based on the pipeline type."""
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
                str(input_file): FileManager.calculate_md5(input_file)
                for input_file in study_inputs.values()
            },
            valid=is_valid,
        )
        FileManager.write_json(hash_outdir / db_id / "info.json", info.model_dump())

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
            PipelineOutputsManager.write_study_results(
                study_dir, study_id, study_results
            )


class IndependentPipeline(Pipeline):
    """Pipeline that processes each study independently."""

    def _process_and_write_study(
        self,
        study_data: Tuple[str, Dict[str, Any], Path],
        hash_outdir: Path,
        input_pipeline_info: Optional[InputPipelineInfo] = None,
        **kwargs,
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

        # Process inputs with StudyInputsManager
        loaded_study_inputs = StudyInputsManager.load_study_inputs(
            study_inputs=study_inputs,
        )

        # Process the inputs with study ID for error tracking
        try:
            results, raw_results, is_valid = self.transform(
                loaded_study_inputs, **kwargs
            )
        except Exception as e:
            logger.error(
                f"Error processing study {db_id}: {e}. "
                f"Inputs: {loaded_study_inputs}"
            )
            return False

        if results:
            # Write results immediately
            self.write_study_info(
                hash_outdir=hash_outdir,
                db_id=db_id,
                study_inputs=study_inputs,
                is_valid=is_valid,
            )
            FileManager.write_json(study_outdir / "results.json", results)

        if raw_results is not results:
            # Write raw results if they differ from cleaned results
            FileManager.write_json(study_outdir / "raw_results.json", results)

        return True

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
            pipeline_info = PipelineOutputsManager.create_pipeline_info(
                extractor_name=self.extractor.__class__.__name__,
                extractor_kwargs={
                    arg: getattr(self, arg)
                    for arg in inspect.signature(self.__init__).parameters.keys()
                },
                version=self.extractor._version,
                config_hash=hash_outdir.name,
                output_schema=self.extractor._output_schema,
                input_pipeline_info=input_pipeline_info,
                transform_kwargs=kwargs,
            )
            PipelineOutputsManager.write_pipeline_info(hash_outdir, pipeline_info)

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
        hash_outdir = self._prepare_pipeline(
            dataset, output_directory, input_pipeline_info
        )

        # Create and write pipeline info at start
        hash_outdir.mkdir(parents=True, exist_ok=True)
        pipeline_info = PipelineOutputsManager.create_pipeline_info(
            extractor_name=self.extractor.__class__.__name__,
            extractor_kwargs={
                arg: getattr(self, arg)
                for arg in inspect.signature(self.__init__).parameters.keys()
            },
            version=self.extractor._version,
            config_hash=hash_outdir.name,
            output_schema=self.extractor._output_schema,
            input_pipeline_info=input_pipeline_info,
            transform_kwargs=kwargs,
        )
        PipelineOutputsManager.write_pipeline_info(hash_outdir, pipeline_info)

        # Check if there are any changes for dependent mode
        if not self.check_for_changes(hash_outdir, dataset, input_pipeline_info):
            logger.info("No changes detected, skipping pipeline execution.")
            return

        # If the directory exists, find the next available directory
        if hash_outdir.exists():
            hash_outdir = FileManager.get_next_available_dir(hash_outdir)
            hash_outdir.mkdir(parents=True, exist_ok=True)

        # Collect all inputs and run the group function at once
        all_study_inputs = {
            db_id: self.collect_study_inputs(study, input_pipeline_info)
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
                                hash_outdir=hash_outdir,
                                db_id=db_id,
                                study_inputs=all_study_inputs[db_id],
                                is_valid=is_valid,
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
        return results
