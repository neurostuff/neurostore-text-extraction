from datetime import datetime
import concurrent.futures
import inspect
import json
import hashlib
import logging
import traceback
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Dict, Any, Union, Optional, Type, Tuple, List
from packaging.version import parse as parse_version

import tqdm
from pydantic import BaseModel

from ns_extract.dataset import Study, Dataset


logger = logging.getLogger(__name__)

# Standard file inputs that come from processed data
INPUTS = [
    "text",
    "coordinates",
    "metadata",
]

# Raw file inputs
RAW_INPUTS = [
    "raw.html",
    "raw.xml",
    "raw.tables",
    "raw.tables_xml",
]

# Pipeline result file types
PIPELINE_INPUTS = [
    "results",
    "raw_results",
    "info",
]


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


class Pipeline(ABC):
    """Abstract pipeline class for processing data."""

    _version: str = None
    _output_schema: Type[BaseModel] = None  # Required schema for output validation

    def __init__(
        self,
        inputs: Union[tuple, list] = ("text",),
        input_sources: tuple = ("pubget", "ace"),
        pipeline_inputs: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize pipeline.

        Args:
            inputs: File inputs from ProcessedData to use
            input_sources: Sources to accept file inputs from
            pipeline_inputs: Dict mapping pipeline names to lists of inputs to use.
                Example: {"participant_demographics": ["results", "info"]}
        """
        if not self._output_schema:
            raise ValueError("Subclass must define _output_schema class")

        self.inputs = inputs
        self.input_sources = input_sources
        self.pipeline_inputs = pipeline_inputs or {}
        self._pipeline_type = (
            inspect.getmro(self.__class__)[1].__name__.lower().rstrip("pipeline")
        )

    def _get_pipeline_version(
        self, pipeline_dir: Path, version_str: str = "latest"
    ) -> Tuple[str, Path]:
        """Get pipeline version directory.

        Args:
            pipeline_dir: Base pipeline directory
            version_str: Version to use or "latest" for highest semver version

        Returns:
            Tuple of (version string, version directory Path)
        """
        if version_str == "latest":
            # Find highest semver version
            version_dirs = [d for d in pipeline_dir.glob("*/") if d.is_dir()]
            if not version_dirs:
                raise ValueError(f"No version directories found in {pipeline_dir}")

            versions = []
            for d in version_dirs:
                try:
                    versions.append(parse_version(d.name))
                except ValueError:
                    logger.warning(f"Invalid version directory name: {d.name}")
                    continue

            if not versions:
                raise ValueError("No valid version directories found")

            version_str = str(max(versions))
            version_dir = pipeline_dir / version_str
        else:
            version_dir = pipeline_dir / version_str
            if not version_dir.exists():
                raise ValueError(f"Version directory not found: {version_dir}")

        return version_str, version_dir

    def _get_pipeline_config(
        self, version_dir: Path, config: str = "latest"
    ) -> Tuple[str, Path, Path]:
        """Get pipeline config directory and info.

        Args:
            version_dir: Pipeline version directory
            config: Config hash to use or "latest" for most recent by date

        Returns:
            Tuple of (config hash, config directory Path, pipeline info Path)
        """
        if config == "latest":
            # Find most recent by pipeline_info.json date
            config_dirs = [d for d in version_dir.glob("*/") if d.is_dir()]
            if not config_dirs:
                raise ValueError(f"No config directories found in {version_dir}")

            latest_date = None
            latest_config = None
            latest_info = None

            for d in config_dirs:
                info_file = d / "pipeline_info.json"
                if not info_file.exists():
                    logger.warning(f"No pipeline_info.json found in {d}")
                    continue

                try:
                    with open(info_file) as f:
                        info = json.load(f)
                    date = datetime.fromisoformat(info["date"])
                    if latest_date is None or date > latest_date:
                        latest_date = date
                        latest_config = d
                        latest_info = info_file
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Error reading pipeline_info.json from {d}: {e}")
                    continue

            if latest_config is None:
                raise ValueError("No valid config directories found")

            config_dir = latest_config
            config_hash = latest_config.name
            pipeline_info = latest_info
        else:
            config_dir = version_dir / config
            if not config_dir.exists():
                raise ValueError(f"Config directory not found: {config_dir}")
            config_hash = config
            pipeline_info = config_dir / "pipeline_info.json"
            if not pipeline_info.exists():
                raise ValueError(f"No pipeline_info.json found in {config_dir}")

        return config_hash, config_dir, pipeline_info

    @abstractmethod
    def transform_dataset(
        self,
        dataset: Dataset,
        output_directory: Union[str, Path],
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Process a full dataset through the pipeline.

        Args:
            dataset: Dataset to process
            output_directory: Directory to write outputs
            input_pipeline_kwargs: Pipeline version/config selection args. Dict mapping:
                pipeline_name -> {
                    "version_str": version_str or "latest",
                    "config_hash": config_hash or "latest",
                    "pipeline_dir": pipeline_dir
                }
                Version and config default to "latest" if not specified.
            **kwargs: Additional arguments
        """
        pass

    def _process_inputs(
        self, study_inputs: Dict[str, Any], study_id: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Process inputs through the full pipeline flow: pre-process, execute, post-process,
        validate.

        Args:
            study_inputs: Dictionary of input data
            study_id: Optional ID of the study being processed (for logging)
            **kwargs: Additional arguments including:

        Returns:
            Dict with:
                - results: Validated results
                - raw_results: Raw results if post-processing was applied
        """
        try:
            # Execute core pipeline logic
            raw_results = self.execute(study_inputs, **kwargs)
            if not raw_results:
                return None

            # Post-process results
            post_results = self.post_process(raw_results, **kwargs)

            results = self.validate_results(
                post_results or raw_results,
                study_id=study_id,
            )

            output = {
                "results": results,
            }

            if post_results:
                output["raw_results"] = raw_results

            return output

        except Exception as e:
            study_info = f" for study {study_id}" if study_id else ""
            logging.error(f"Pipeline execution failed{study_info}: {e}")
            logging.error(f"Full Traceback: {traceback.format_exc()}")
            return None

    def validate_results(self, results: dict, **kwargs) -> Optional[dict]:
        """Validate results against the output schema.

        Args:
            results: Raw or post-processed results from pipeline
            **kwargs: Additional arguments including:
                - study_id: Optional ID of the study being validated (for logging)

        Returns:
            Tuple of (is_valid, results)
        """
        study_id = kwargs.get("study_id")
        try:
            validated = self._output_schema.model_validate(results)
            return True, validated.model_dump()
        except Exception as e:
            study_info = f" for study {study_id}" if study_id else ""
            logging.error(f"Raw result validation error{study_info}: {e}")
            return False, results

    def post_process(self, results: dict, **kwargs) -> dict:
        """Post-process results before validation. Override in subclass if needed.

        Args:
            results: Raw results from pipeline
            **kwargs: Additional arguments including:
                - study_id: Optional ID of the study being validated (for logging)
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

    def create_directory_hash(
        self, dataset: Dataset, output_directory: Path
    ) -> Tuple[Path, str]:
        """Create a hash for the dataset."""
        dataset_str = self._serialize_dataset_keys(dataset)
        arg_str = self._serialize_pipeline_args()
        if dataset.pipelines:
            input_pipelines_str = '_'.join([
                f"{pipeline.name}_{pipeline.version}_{pipeline.config_hash}"
                for pipeline in dataset.pipelines.values()])
            full_str = f"{dataset_str}_{input_pipelines_str}_{arg_str}"
        else:
            full_str = f"{dataset_str}_{arg_str}"
        hash_str = hashlib.shake_256(full_str.encode()).hexdigest(6)
        outdir = output_directory / self.__class__.__name__ / self._version / hash_str
        return outdir, hash_str

    def filter_inputs(self, output_directory: Path, dataset: Dataset) -> bool:
        """Filter inputs based on the pipeline type."""
        existing_results = self._filter_existing_results(output_directory, dataset)
        matching_results = self._identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        keep_ids = set(dataset.data.keys()) - {
            db_id for db_id, match in matching_results.items() if match
        }
        return dataset.slice(keep_ids)

    def gather_all_study_inputs(
        self,
        dataset: Dataset,
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Dict[str, Path]]:
        """Collect all inputs for the dataset.

        Args:
            dataset: Dataset to collect inputs from
            input_pipeline_kwargs: Optional pipeline version/config selection arguments
        """
        return {
            db_id: self.collect_study_inputs(study, input_pipeline_kwargs)
            for db_id, study in dataset.data.items()
        }

    def collect_study_inputs(
        self,
        study: Study,
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Path]:
        """Collect inputs for a study.

        Args:
            study: Study object to get inputs from
            input_pipeline_kwargs: Optional pipeline arguments

        Returns:
            Dict mapping input names to file paths
        """
        study_inputs = {}

        # Collect file inputs from sources
        for source in self.input_sources:
            # Skip pipeline sources - handled below
            if source in self.pipeline_inputs:
                continue

            source_obj = getattr(study, source, None)
            if source_obj:
                for input_type in self.inputs:
                    # Skip if not a regular file input
                    if input_type in PIPELINE_INPUTS:
                        continue
                    input_obj = deep_getattr(source_obj, input_type, None)
                    if input_obj and study_inputs.get(input_type) is None:
                        study_inputs[input_type] = input_obj

        # Get required pipeline inputs using pipeline_results
        for pipeline_name, required_inputs in self.pipeline_inputs.items():
            if input_pipeline_kwargs and pipeline_name in input_pipeline_kwargs:
                pipeline_kwargs = input_pipeline_kwargs[pipeline_name]
                pipeline_dir = Path(pipeline_kwargs["pipeline_directory"])
                version = pipeline_kwargs["version"]
                config_hash = pipeline_kwargs["config_hash"]

                # Validate pipeline directory structure
                if not pipeline_dir.exists():
                    raise ValueError(
                        f"Pipeline directory does not exist: {pipeline_dir}"
                    )

                has_version_dirs = any(d.is_dir() for d in pipeline_dir.glob("*"))
                if not has_version_dirs:
                    raise ValueError("No version directories found in pipeline directory")

                # Validate version and config dirs
                version_dir = pipeline_dir / version
                if not version_dir.exists():
                    raise ValueError(f"Version directory does not exist: {version}")

                config_dir = version_dir / config_hash
                if not config_dir.exists():
                    raise ValueError(f"Config directory does not exist: {config_hash}")

                # Check for study results
                results_file = config_dir / study.dbid / "results.json"
                if not results_file.exists():
                    raise ValueError("Missing results.json file")

                study_inputs[pipeline_name] = results_file
            else:
                # Get pipeline results if no kwargs provided
                pipeline_result = study.get_pipeline_result(pipeline_name)
                if not pipeline_result:
                    raise ValueError(
                        f"Missing pipeline results for {pipeline_name} pipeline "
                        f"(study {study.dbid})"
                    )
                study_inputs[pipeline_name] = pipeline_result.result

            # Validate required input types against allowed types
            for input_type in required_inputs:
                if input_type not in PIPELINE_INPUTS:
                    raise ValueError(
                        f"Invalid pipeline input type: {input_type}. "
                        f"Must be one of {PIPELINE_INPUTS}"
                    )

        return study_inputs

    def write_pipeline_info(
        self,
        hash_outdir: Path,
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None,
        transform_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Write information about the pipeline to a pipeline_info.json file.
        
        Args:
            hash_outdir: Directory to write pipeline info
            input_pipeline_kwargs: Pipeline kwargs passed to transform_dataset
            transform_kwargs: Additional kwargs passed to transform_dataset
        """
        # Get output schema fields and their types
        output_schema = {}
        if self._output_schema:
            output_schema = {
                field: str(field_info.annotation)
                for field, field_info in self._output_schema.model_fields.items()
            }

        pipeline_info = {
            "date": datetime.now().isoformat(),
            "version": self._version,
            "type": self._pipeline_type,
            "arguments": {
                arg: getattr(self, arg)
                for arg in inspect.signature(self.__init__).parameters.keys()
            },
            "output_schema": output_schema,
            "input_pipeline_kwargs": input_pipeline_kwargs or {},
            "transform_kwargs": transform_kwargs or {},
        }
        FileManager.write_json(hash_outdir / "pipeline_info.json", pipeline_info)

    def write_study_info(
        self,
        hash_outdir: Path,
        db_id: str,
        study_inputs: Dict[str, Path],
        is_valid: bool,
    ):
        """Write information about the current run to an info.json file."""
        output_info = {
            "date": datetime.now().isoformat(),
            "inputs": {
                str(input_file): FileManager.calculate_md5(input_file)
                for input_file in study_inputs.values()
            },
            "valid": is_valid,
        }
        FileManager.write_json(hash_outdir / db_id / "info.json", output_info)

    def _serialize_dataset_keys(self, dataset: Dataset) -> str:
        """Return a hashable string of the input dataset."""
        return "_".join(list(dataset.data.keys()))

    def _serialize_pipeline_args(self) -> str:
        """Return a hashable string of the arguments."""
        args = list(inspect.signature(self.__init__).parameters.keys())
        return "_".join([f"{arg}_{str(getattr(self, arg))}" for arg in args])

    def _filter_existing_results(
        self, output_dir: Path, dataset: Dataset
    ) -> Dict[str, Dict]:
        """Find the most recent result for an existing study."""
        existing_results = {}
        result_directory = output_dir / self.__class__.__name__ / self._version
        current_args = {
            arg: getattr(self, arg)
            for arg in inspect.signature(self.__init__).parameters.keys()
        }

        current_args = json.loads(json.dumps(current_args))

        for d in result_directory.glob("*"):
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

    def _identify_matching_results(
        self,
        dataset: Dataset,
        existing_results: Dict[str, Dict],
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, bool]:
        """Compare dataset inputs with existing results."""
        dataset_inputs = self.gather_all_study_inputs(
            dataset=dataset,
            input_pipeline_kwargs=input_pipeline_kwargs
        )
        return {
            db_id: self._are_file_hashes_identical(
                study_inputs, existing_results.get(db_id, {}).get("inputs", {})
            )
            for db_id, study_inputs in dataset_inputs.items()
        }


class IndependentPipeline(Pipeline):
    """Pipeline that processes each study independently."""

    def create_directory_hash(
        self, dataset: Dataset, output_directory: Path
    ) -> Tuple[Path, str]:
        """Create a hash for independent pipeline execution.

        For independent pipelines, the hash is based only on:
        1. Pipeline arguments
        2. Input pipeline versions/configs (if any)

        The dataset study IDs are not included since each study is processed independently.
        """
        arg_str = self._serialize_pipeline_args()
        if dataset.pipelines:
            input_pipelines_str = '_'.join([
                f"{pipeline.name}_{pipeline.version}_{pipeline.config_hash}"
                for pipeline in dataset.pipelines.values()])
            full_str = f"{input_pipelines_str}_{arg_str}"
        else:
            full_str = arg_str

        hash_str = hashlib.shake_256(full_str.encode()).hexdigest(6)
        outdir = output_directory / self.__class__.__name__ / self._version / hash_str
        return outdir, hash_str

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
        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        if input_pipeline_kwargs:
            for pipeline_name, pipeline_kwargs in input_pipeline_kwargs.items():
                # Add pipeline with standardized argument names
                dataset.add_pipeline(
                    name=pipeline_name,
                    version=pipeline_kwargs["version"],
                    config_hash=pipeline_kwargs["config_hash"],
                    pipeline_directory=pipeline_kwargs["pipeline_directory"]
                )

        hash_outdir, hash_str = self.create_directory_hash(dataset, output_directory)

        if not hash_outdir.exists():
            hash_outdir.mkdir(parents=True)
            # Include transform arguments in pipeline info
            transform_kwargs = {"num_workers": num_workers, **kwargs}
            self.write_pipeline_info(
                hash_outdir,
                input_pipeline_kwargs=input_pipeline_kwargs,
                transform_kwargs=transform_kwargs
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

        with tqdm.tqdm(
            total=len(studies_to_process), desc="Processing studies"
        ) as pbar:
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

    def check_for_changes(
        self,
        output_directory: Path,
        dataset: Dataset,
        input_pipeline_kwargs: Optional[Dict[str, Dict[str, str]]] = None
    ) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self._filter_existing_results(output_directory, dataset)
        matching_results = self._identify_matching_results(
            dataset,
            existing_results,
            input_pipeline_kwargs=input_pipeline_kwargs
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
        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        if input_pipeline_kwargs:
            for pipeline_name, pipeline_kwargs in input_pipeline_kwargs.items():
                # Add pipeline with standardized argument names
                dataset.add_pipeline(
                    name=pipeline_name,
                    version=pipeline_kwargs["version"],
                    config_hash=pipeline_kwargs["config_hash"],
                    pipeline_directory=pipeline_kwargs["pipeline_directory"]
                )

        hash_outdir, hash_str = self.create_directory_hash(dataset, output_directory)

        # Check if there are any changes for dependent mode
        if not self.check_for_changes(output_directory, dataset, input_pipeline_kwargs):
            print("No changes detected, skipping pipeline execution.")
            return  # No changes, so we skip the pipeline

        # If the directory exists, find the next available directory
        # with a suffix like "-1", "-2", etc.
        if hash_outdir.exists():
            hash_outdir = FileManager.get_next_available_dir(hash_outdir)
        hash_outdir.mkdir(parents=True, exist_ok=True)
        self.write_pipeline_info(
            hash_outdir,
            input_pipeline_kwargs=input_pipeline_kwargs,
            transform_kwargs=kwargs
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
