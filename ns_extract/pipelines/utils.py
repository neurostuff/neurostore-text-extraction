"""Utility classes and functions for pipeline operations.

This module provides mixins implementing common functionality for:
- File operations (reading, writing, hashing)
- Study input handling (loading and validating study data)
- Pipeline output management (metadata and results handling)

The mixins are designed to be composable and reusable across different
pipeline implementations.
"""

import hashlib
import inspect
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional, List

from ns_extract.pipelines.exceptions import FileOperationError
from ns_extract.pipelines.data_structures import (
    InputPipelineInfo,
    PipelineOutputInfo,
    StudyOutputJson,
)

logger = logging.getLogger(__name__)


class FileOperationsMixin:
    """Mixin providing common file operation methods.

    Implements core file system operations used throughout the pipeline:
    - File hashing for change detection
    - JSON serialization/deserialization
    - Directory management
    """

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file's contents.

        Args:
            file_path: Path to file to hash

        Returns:
            Hexadecimal string of MD5 hash

        Raises:
            IOError: If file cannot be read
        """
        with file_path.open("r") as f:
            file_contents = f.read()
        return hashlib.md5(file_contents.encode()).hexdigest()

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse JSON from a file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON content as dictionary

        Raises:
            IOError: If file cannot be read
            json.JSONDecodeError: If file contains invalid JSON
        """
        with file_path.open("r") as f:
            return json.load(f)

    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write data to a JSON file.

        Args:
            file_path: Output file path
            data: Data to serialize to JSON

        Raises:
            IOError: If file cannot be written
            TypeError: If data contains non-serializable types
        """
        with file_path.open("w") as f:
            json.dump(data, f, default=str, indent=4)

    def _find_unique_directory(self, base_path: Path) -> Path:
        """Find next available directory name by appending incrementing numbers.

        Used to create uniquely named directories when a target directory
        already exists. Appends -1, -2, etc. until an unused name is found.

        Args:
            base_path: Initial desired directory path

        Returns:
            Path with number appended if needed to make it unique

        Example:
            >>> get_next_available_dir(Path('results'))
            Path('results')  # if 'results' doesn't exist
            >>> get_next_available_dir(Path('results'))
            Path('results-1')  # if 'results' exists
        """
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(f"{base_path.name}-{counter}")
            counter += 1
        return new_path


class StudyInputsMixin:
    """Mixin providing study input handling methods.

    This mixin handles loading and preprocessing of study input files of various types:
    - Text files (.txt)
    - JSON files (.json)
    - CSV files (.csv)

    It provides error handling and consistent interfaces for working with
    different file formats.
    """

    def _load_text_file(self, file_path: Path) -> str:
        """Read text file contents with robust error handling.

        Args:
            file_path: Path to text file to read

        Returns:
            Contents of file as string

        Raises:
            IOError: If file cannot be read, with detailed error message
        """
        try:
            with file_path.open("r") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read text file {file_path}: {str(e)}")

    def _load_study_inputs(
        self, study_inputs: Dict[str, Union[str, Path]]
    ) -> Dict[str, Union[str, Dict[str, Any], List[Dict[str, Any]]]]:
        """Load and parse study input files based on their extensions.

        Handles multiple file types:
        - .txt: Returns contents as string
        - .json: Returns parsed JSON as dict
        - .csv: Returns list of row dicts using pandas

        Args:
            study_inputs: Dict mapping input names to file paths
                Format: {"input_name": "path/to/file"}

        Returns:
            Dict mapping input names to loaded contents
            Format varies by file type:
            - txt: {"input": "text content"}
            - json: {"input": {parsed json}}
            - csv: {"input": [{row1}, {row2}, ...]}

        Raises:
            IOError: If any file cannot be read/parsed
            ValueError: If file has unsupported extension
        """
        loaded_inputs: Dict[str, Union[str, Dict[str, Any], List[Dict[str, Any]]]] = {}
        for input_name, file_path in study_inputs.items():
            path = Path(file_path)
            suffix = path.suffix.lower()

            try:
                if suffix == ".txt":
                    loaded_inputs[input_name] = self._load_text_file(path)
                elif suffix == ".json":
                    loaded_inputs[input_name] = self._load_json(path)
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


class PipelineOutputsMixin(FileOperationsMixin):
    """Mixin providing pipeline output handling methods.

    This mixin handles the management of pipeline outputs including:
    - Pipeline metadata serialization and storage
    - Study results writing and validation
    - Input pipeline dependency tracking

    It ensures consistent output formatting and provides robust error handling
    for all file operations.
    """

    def _normalize_pipeline_info(
        self,
        info: Optional[
            Union[Dict[str, Dict[str, str]], Dict[str, "InputPipelineInfo"]]
        ],
    ) -> Dict[str, "InputPipelineInfo"]:
        """Convert pipeline info to strongly-typed objects.

        Args:
            info: Dict mapping pipeline names to either:
                - Configuration parameters as dict
                - InputPipelineInfo instances directly

        Returns:
            Dict mapping names to InputPipelineInfo instances

        Note:
            Returns empty dict if info is None
        """
        from ns_extract.pipelines.data_structures import InputPipelineInfo

        if info is None:
            return {}

        result = {}
        for name, value in info.items():
            if isinstance(value, InputPipelineInfo):
                result[name] = value
            else:
                result[name] = InputPipelineInfo(**value)
        return result

    def _create_pipeline_info(
        self,
        hash_outdir: Path,
        transform_kwargs: Dict[str, Any],
        input_pipelines: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> PipelineOutputInfo:
        """Create pipeline metadata object with configuration details."""
        return PipelineOutputInfo(
            date=datetime.now().isoformat(),
            version=self.extractor._version,
            config_hash=hash_outdir.name,
            extractor=self.extractor.__class__.__name__,
            extractor_kwargs={
                arg: getattr(self, arg)
                for arg in inspect.signature(self.__init__).parameters.keys()
            },
            transform_kwargs=transform_kwargs,
            input_pipelines=input_pipelines or {},
            schema=self.extractor._output_schema.model_json_schema(),
        )

    def _write_pipeline_info(
        self, hash_outdir: Path, info: "PipelineOutputInfo"
    ) -> None:
        """Write pipeline metadata to pipeline_info.json.

        Args:
            hash_outdir: Directory to write metadata file into
            info: Pipeline metadata object to serialize

        Raises:
            ValueError: If hash_outdir is not a Path object
            IOError: If file cannot be written
        """
        if not isinstance(hash_outdir, Path):
            raise ValueError("output_dir must be a Path object")

        try:
            info_path = hash_outdir / "pipeline_info.json"
            self._write_json(info_path, info.model_dump())
        except IOError as e:
            logger.error(f"Failed to write pipeline info: {str(e)}")
            raise

    def _write_study_info(
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
                str(input_file): self._calculate_md5(input_file)
                for input_file in study_inputs.values()
            },
            valid=is_valid,
        )
        self._write_json(hash_outdir / db_id / "info.json", info.model_dump())

    def _write_study_results(
        self,
        study_dir: Path,
        study_id: str,
        cleaned_results: Dict[str, Any],
        raw_results: Dict[str, Any],
    ) -> None:
        """Write study results to output directory.

        Writes two potential output files:
        - results.json: Final processed results
        - raw_results.json: Only written when different from cleaned_results

        Args:
            study_dir: Directory to write results into (assumed to exist)
            study_id: Identifier of the study being processed
            cleaned_results: Final processed results to write
            raw_results: Raw results to write if different from cleaned

        Raises:
            FileOperationError: If file writing fails
        """
        try:
            # Write raw results if different from cleaned
            if raw_results != cleaned_results:
                self._write_json(study_dir / "raw_results.json", raw_results)

            # Write cleaned results
            self._write_json(study_dir / "results.json", cleaned_results)

        except IOError as e:
            raise FileOperationError(
                f"Failed to write results for study {study_id}: {str(e)}"
            )
