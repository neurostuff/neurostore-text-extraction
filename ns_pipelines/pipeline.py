from datetime import datetime
import json
import hashlib
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Dict, Any, List, Union


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

    _hash_attrs: List[str] = ['_inputs', '_input_sources']
    _pipeline_type: str = None  # independent or dependent
    _version: str = None

    def __init__(self, inputs: Union[tuple, list] = ("text",), input_sources: tuple = ("pubget", "ace")):
        self._inputs = inputs
        self._input_sources = input_sources

    def serialize_dataset_keys(self, dataset: Any) -> str:
        """Return a hashable string of the input dataset."""
        return "_".join(list(dataset.data.keys()))

    def serialize_pipeline_args(self) -> str:
        """Return a hashable string of the arguments."""
        return '_'.join([str(getattr(self, arg)) for arg in self._hash_attrs])

    def full_output_hash(self, dataset_str: str, arg_str: str) -> str:
        """Return the full hash."""
        return hashlib.shake_256(f"{dataset_str}_{arg_str}".encode()).hexdigest(6)

    def create_directory_hash(self, dataset: Any) -> str:
        """Create a hash for the dataset."""
        dataset_str = self.serialize_dataset_keys(dataset)
        arg_str = self.serialize_pipeline_args()
        return self.full_output_hash(dataset_str, arg_str)

    def filter_existing_results(self, output_dir: Path, dataset: Any) -> Dict[str, Dict]:
        """Find the most recent result for an existing study."""
        existing_results = {}
        for d in output_dir.glob(f"{self._version}/**/*"):
            if d.is_dir() and d.name in set(dataset.data.keys()):
                info_file = d / "info.json"
                if info_file.exists():
                    info = FileManager.load_json(info_file)
                    found_info = {
                        "date": info["date"],
                        "inputs": info["inputs"],
                        "hash": d.parents[0].name
                    }
                    if (
                        existing_results.get(d.name) is None
                        or datetime.strptime(info["date"], '%Y-%m-%d') >
                        datetime.strptime(existing_results[d.name]["date"], '%Y-%m-%d')
                    ):
                        existing_results[d.name] = found_info
        return existing_results

    def are_file_hashes_identical(self, study_inputs: Dict[str, Path], existing_inputs: Dict[str, str]) -> bool:
        """Compare file hashes to determine if the inputs have changed."""
        if set(str(p) for p in study_inputs.values()) != set(existing_inputs.keys()):
            return False

        for existing_file, hash_val in existing_inputs.items():
            if FileManager.calculate_md5(Path(existing_file)) != hash_val:
                return False

        return True

    def collect_study_inputs(self, study: Any) -> Dict[str, Path]:
        """Collect inputs for a study."""
        study_inputs = {}
        for source in self._input_sources:
            source_obj = getattr(study, source, None)
            if source_obj:
                for input_type in self._inputs:
                    input_obj = deep_getattr(source_obj, input_type, None)
                    if input_obj and study_inputs.get(input_type) is None:
                        study_inputs[input_type] = input_obj
        return study_inputs

    def gather_all_study_inputs(self, dataset: Any) -> Dict[str, Dict[str, Path]]:
        """Collect all inputs for the dataset."""
        return {db_id: self.collect_study_inputs(study) for db_id, study in dataset.data.items()}

    def identify_matching_results(self, dataset: Any, existing_results: Dict[str, Dict]) -> Dict[str, bool]:
        """Compare dataset inputs with existing results."""
        dataset_inputs = self.gather_all_study_inputs(dataset)
        return {
            db_id: self.are_file_hashes_identical(study_inputs, existing_results.get(db_id, {}).get("inputs", {}))
            for db_id, study_inputs in dataset_inputs.items()
        }

    def filter_inputs(self, output_directory: Path, dataset: Any) -> bool:
        """Filter inputs based on the pipeline type."""
        existing_results = self.filter_existing_results(output_directory, dataset)
        matching_results = self.identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        keep_ids = set(dataset.data.keys()) - {db_id for db_id, match in matching_results.items() if match}
        return dataset.slice(keep_ids)

    def check_for_changes(self, output_directory: Path, dataset: Any) -> bool:
        """Check if any study inputs have changed or if there are new studies."""
        existing_results = self.filter_existing_results(output_directory, dataset)
        matching_results = self.identify_matching_results(dataset, existing_results)
        # Return True if any of the studies' inputs have changed or if new studies exist
        return any(not match for match in matching_results.values())

    def write_output_info(self, hash_outdir: Path, db_id: str, study_inputs: Dict[str, Path]):
        """Write information about the current run to an info.json file."""
        output_info = {
            "date": datetime.now().isoformat(),
            "inputs": {str(input_file): FileManager.calculate_md5(input_file) for input_file in study_inputs.values()}
        }
        FileManager.write_json(hash_outdir / db_id / "info.json", output_info)

    @abstractmethod
    def run(self, dataset: Any, output_directory: Path):
        """Run the pipeline."""
        pass

    @abstractmethod
    def function(self, study_inputs: Dict[str, Any]) -> Dict:
        """Run the pipeline function."""
        pass


class IndependentPipeline(Pipeline):
    """Pipeline that processes each study independently."""

    _pipeline_type = "independent"

    def run(self, dataset: Any, output_directory: Path):
        """Run the pipeline for independent studies."""
        hash_str = self.create_directory_hash(dataset)
        hash_outdir = output_directory / self._version / hash_str

        # If the directory exists, find the next available directory with a suffix like "-1", "-2", etc.
        if hash_outdir.exists():
            hash_outdir = FileManager.get_next_available_dir(hash_outdir)
        hash_outdir.mkdir(parents=True, exist_ok=True)

        # Process each study individually
        filtered_dataset = self.filter_inputs(output_directory, dataset)
        for db_id, study in filtered_dataset.data.items():
            study_inputs = self.collect_study_inputs(study)
            study_outdir = hash_outdir / db_id
            study_outdir.mkdir(parents=True, exist_ok=True)

            results = self.function(study_inputs)
            FileManager.write_json(study_outdir / "results.json", results)

            self.write_output_info(hash_outdir, db_id, study_inputs)


class DependentPipeline(Pipeline):
    """Pipeline that processes all studies as a group."""

    _pipeline_type = "dependent"

    def run(self, dataset: Any, output_directory: Path):
        """Run the pipeline for dependent studies."""
        hash_str = self.create_directory_hash(dataset)
        hash_outdir = output_directory / self._version / hash_str

        # Check if there are any changes for dependent mode
        if not self.check_for_changes(output_directory, dataset):
            print("No changes detected, skipping pipeline execution.")
            return  # No changes, so we skip the pipeline

        # If the directory exists, find the next available directory with a suffix like "-1", "-2", etc.
        if hash_outdir.exists():
            hash_outdir = FileManager.get_next_available_dir(hash_outdir)
        hash_outdir.mkdir(parents=True, exist_ok=True)

        # Collect all inputs and run the group function at once
        all_study_inputs = self.gather_all_study_inputs(dataset)
        grouped_results = self.function(all_study_inputs)
        for db_id, results in grouped_results.items():
            study_outdir = hash_outdir / db_id
            study_outdir.mkdir(parents=True, exist_ok=True)
            FileManager.write_json(study_outdir / "results.json", results)
            self.write_output_info(hash_outdir, db_id, all_study_inputs[db_id])
