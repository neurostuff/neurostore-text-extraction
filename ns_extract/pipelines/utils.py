"""Utility classes and functions for pipeline operations."""

import hashlib
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FileOperationsMixin:
    """Mixin providing common file operation methods."""

    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        with file_path.open("r") as f:
            file_contents = f.read()
        return hashlib.md5(file_contents.encode()).hexdigest()

    def load_json(self, file_path: Path) -> Dict:
        """Load JSON from a file."""
        with file_path.open("r") as f:
            return json.load(f)

    def write_json(self, file_path: Path, data: Dict):
        """Write JSON to a file."""
        with file_path.open("w") as f:
            json.dump(data, f)

    def get_next_available_dir(self, base_path: Path) -> Path:
        """Find the next available directory by appending numbers."""
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(f"{base_path.name}-{counter}")
            counter += 1
        return new_path


class StudyInputsMixin:
    """Mixin providing study input handling methods."""

    def load_text_file(self, file_path: Path) -> str:
        """Read text file with error handling."""
        try:
            with file_path.open("r") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read text file {file_path}: {str(e)}")

    def load_study_inputs(self, study_inputs: Dict[str, Union[str, Path]]) -> Dict[str, Any]:
        """Load study input files based on their file extensions."""
        loaded_inputs = {}
        for input_name, file_path in study_inputs.items():
            path = Path(file_path)
            suffix = path.suffix.lower()

            try:
                if suffix == ".txt":
                    loaded_inputs[input_name] = self.load_text_file(path)
                elif suffix == ".json":
                    loaded_inputs[input_name] = self.load_json(path)
                elif suffix == ".csv":
                    import pandas as pd
                    loaded_inputs[input_name] = pd.read_csv(path).to_dict("records")
                else:
                    raise ValueError(f"Unsupported file type for {input_name}: {suffix}")
            except Exception as e:
                raise IOError(f"Failed to load study input {input_name} from {path}: {str(e)}")

        return loaded_inputs


class PipelineOutputsMixin:
    """Mixin providing pipeline output handling methods."""

    def convert_pipeline_info(self, info: Dict[str, Dict[str, str]]) -> Dict:
        """Convert pipeline info dict to InputPipelineInfo objects."""
        from ns_extract.pipelines.base import InputPipelineInfo

        if info is None:
            return {}

        return {name: InputPipelineInfo(**kwargs) for name, kwargs in info.items()}

    def create_pipeline_info(
        self,
        extractor_name: str,
        extractor_kwargs: Dict[str, Any],
        version: str,
        config_hash: str,
        output_schema: Type[BaseModel],
        input_pipeline_info: Optional[Dict[str, Dict[str, str]]] = None,
        transform_kwargs: Dict[str, Any] = None,
    ) -> BaseModel:
        """Create pipeline info instance with provided data."""
        from ns_extract.pipelines.base import PipelineOutputInfo, InputPipelineInfo

        if input_pipeline_info is None:
            input_pipelines = {}
        else:
            input_pipelines = {
                name: InputPipelineInfo(**kwargs)
                for name, kwargs in input_pipeline_info.items()
            }

        return PipelineOutputInfo(
            date=datetime.now().isoformat(),
            version=version,
            config_hash=config_hash,
            extractor=extractor_name,
            extractor_kwargs=extractor_kwargs or {},
            transform_kwargs=transform_kwargs or {},
            input_pipelines=input_pipelines,
            schema=output_schema.model_json_schema(),
        )

    def write_pipeline_info(self, hash_outdir: Path, info: BaseModel) -> None:
        """Write pipeline metadata to pipeline_info.json."""
        if not isinstance(hash_outdir, Path):
            raise ValueError("output_dir must be a Path object")

        try:
            info_path = hash_outdir / "pipeline_info.json"
            self.write_json(info_path, info.model_dump())
        except IOError as e:
            logger.error(f"Failed to write pipeline info: {str(e)}")
            raise

    def write_study_results(self, study_dir: Path, study_id: str, results: Dict[str, Any]) -> None:
        """Write study results to directory."""
        try:
            study_dir.mkdir(exist_ok=True)

            # Write raw results if provided
            if "raw_results" in results:
                raw_path = study_dir / "raw_results.json"
                self.write_json(raw_path, results["raw_results"])

            # Write final results
            results_path = study_dir / "results.json"
            final_results = results.get("results", results)
            self.write_json(results_path, final_results)

        except IOError as e:
            raise IOError(f"Failed to write results for study {study_id}: {str(e)}")

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate that results have the expected structure."""
        if not isinstance(results, dict):
            return False

        # Case 1: Direct results dict
        if all(isinstance(v, (dict, list, str, int, float, bool)) for v in results.values()):
            return True

        # Case 2: Results with optional raw_results
        if "results" in results:
            if not isinstance(results["results"], dict):
                return False
            if "raw_results" in results and not isinstance(results["raw_results"], dict):
                return False
            return True

        return False
