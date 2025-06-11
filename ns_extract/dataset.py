"""Dataset creation for processing inputs."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Union, Optional, Dict
import logging
from datetime import datetime
import json
from packaging.version import parse as parse_version

logger = logging.getLogger(__name__)

INPUTS = [
    "text",
    "coordinates",
    "metadata",
    "html",
    "xml",
    "tables",
    "tables_xml",
]


@dataclass
class PipelineInfo:
    """Pipeline metadata."""

    name: str
    version: str
    config_hash: str
    pipeline_info: Path


@dataclass
class PipelineRunResult:
    """Result from an pipeline run."""

    pipeline: PipelineInfo
    result: Path
    info: Path
    raw_result: Optional[Path] = None

    def __post_init__(self):
        # Validate paths exist
        if not self.result.exists():
            raise ValueError(f"Result file does not exist: {self.result}")
        if self.raw_result and not self.raw_result.exists():
            raise ValueError(f"Raw result file does not exist: {self.raw_result}")
        if not self.info.exists():
            raise ValueError(f"Info file does not exist: {self.info}")


@dataclass
class AceRaw:
    html: Path

    def __post_init__(self):
        # Convert string path to Path object
        # Preprocessing logic for AceRaw can be added here if needed
        if not self.html.exists():
            raise ValueError(f"HTML file {self.html} does not exist.")


@dataclass
class PubgetRaw:
    xml: Path
    tables: dict = field(default_factory=dict)
    tables_xml: Path = None

    def __post_init__(self):
        # Load tables and assign file paths
        if not self.xml.exists():
            raise ValueError(f"XML file {self.xml} does not exist.")

        if self.tables_xml and not self.tables_xml.exists():
            raise ValueError(f"Tables XML file {self.tables_xml} does not exist.")

        if self.tables_xml:
            tables_files = list(self.tables_xml.parent.glob("*.xml"))
            tables_files = [t for t in tables_files if t.name != self.tables_xml.name]

            num_tables = len(tables_files) // 2
            self.tables = {
                f"{t:03}": {"metadata": None, "contents": None}
                for t in range(num_tables)
            }

            for tf in tables_files:
                table_number = tf.stem.split("_")[1]
                if tf.suffix == ".json":
                    key = "metadata"
                else:
                    key = "contents"
                self.tables[table_number][key] = tf


@dataclass
class ProcessedData:
    coordinates: Optional[Path] = None
    text: Path = None
    metadata: Optional[Path] = None
    raw: Optional[Union["PubgetRaw", "AceRaw"]] = field(default=None)

    def __post_init__(self):
        # Ensure the processed data files exist
        if self.coordinates and not self.coordinates.exists():
            raise ValueError(f"Coordinates file {self.coordinates} does not exist.")
        if self.text and not self.text.exists():
            raise ValueError(f"Text file {self.text} does not exist.")
        if self.metadata and not self.metadata.exists():
            raise ValueError(f"Metadata file {self.metadata} does not exist.")


@dataclass
class Study:
    study_dir: Path
    dbid: str = None
    doi: str = None
    pmid: str = None
    pmcid: str = None
    ace: ProcessedData = None
    pubget: ProcessedData = None
    pipeline_results: Dict[str, PipelineRunResult] = field(default_factory=dict)

    def __post_init__(self):
        self.dbid = self.study_dir.name

        # Load identifiers
        with open((self.study_dir / "identifiers.json"), "r") as ident_fp:
            identifiers = json.load(ident_fp)

        self.pmid = identifiers.get("pmid")
        self.pmcid = identifiers.get("pmcid")
        self.doi = identifiers.get("doi")

        # Setup the processed data objects
        # Load AceRaw if available
        source_dir = self.study_dir / "source"
        ace_raw = None
        pubget_raw = None

        # Load AceRaw if available
        ace_path = source_dir / "ace" / f"{self.pmid}.html"
        if ace_path.exists():
            ace_raw = AceRaw(html=ace_path)

        # Load PubgetRaw if available
        pubget_dir = source_dir / "pubget"
        pubget_xml_path = pubget_dir / f"{self.pmcid}.xml"
        tables_xml_path = pubget_dir / "tables" / "tables.xml"
        if pubget_xml_path.exists():
            pubget_raw = PubgetRaw(xml=pubget_xml_path, tables_xml=tables_xml_path)

        # Load processed data
        for t in ["ace", "pubget"]:
            processed_dir = self.study_dir / "processed" / t
            if processed_dir.exists():
                try:
                    processed = ProcessedData(
                        coordinates=processed_dir / "coordinates.csv",
                        text=processed_dir / "text.txt",
                        metadata=processed_dir / "metadata.json",
                        raw=ace_raw if t == "ace" else pubget_raw,
                    )
                except ValueError as e:
                    logger.error(f"Error loading processed data for {self.dbid}: {e}")
                    continue

                setattr(self, t, processed)

    def add_pipeline_result(self, result: PipelineRunResult):
        """Add an pipeline result to the study.

        Args:
            result: PipelineRunResult containing paths to results and metadata
        """
        self.pipeline_results[result.pipeline.name] = result

    def get_pipeline_result(self, pipeline_name: str) -> Optional[PipelineRunResult]:
        """Get am pipeline result by pipeline name.

        Args:
            pipeline_name: Name of the pipeline

        Returns:
            PipelineRunResult if found, None otherwise
        """
        return self.pipeline_results.get(pipeline_name)


class Dataset:
    """Dataset class for processing inputs."""

    def __init__(self, input_directory):
        """Initialize the dataset."""
        self.data = self.load_directory(input_directory)
        self.pipelines = {}

    def slice(self, ids):
        """Slice the dataset."""
        deepcopy_obj = deepcopy(self)
        deepcopy_obj.data = {k: v for k, v in deepcopy_obj.data.items() if k in ids}
        return deepcopy_obj

    def load_directory(self, input_directory):
        """Load the input directory."""
        if isinstance(input_directory, str):
            input_directory = Path(input_directory)

        # always resolve to absolute path
        input_directory = input_directory.resolve()
        if not input_directory.exists():
            raise ValueError(f"Input directory {input_directory} does not exist.")

        pattern = re.compile(r"^[a-zA-Z0-9]{12}$")
        sub_directories = input_directory.glob("[0-9A-Za-z]*")
        study_directories = [
            dir_
            for dir_ in sub_directories
            if dir_.is_dir() and pattern.match(dir_.name)
        ]

        dset_data = {}

        for study_dir in study_directories:
            study_obj = Study(study_dir=study_dir)

            dset_data[study_obj.dbid] = study_obj

        if not dset_data:
            raise ValueError(f"No valid studies found in {input_directory}")

        return dset_data

    def add_pipeline(
        self,
        pipeline_name: str,
        pipeline_dir: Union[str, Path],
        version: str = "latest",
        config_hash: str = "latest",
    ):
        """Add pipeline results to studies in the dataset.

        Args:
            pipeline_name: Name of the pipeline (e.g., "UMLSDiseasePipeline")
            pipeline_dir: Base directory containing all pipeline results in structure:
                <pipeline_dir>/<pipeline_name>/<version>/<config_hash>/
            version_str: Version to use, or "latest" to use highest semver version
            config: Config hash to use, or "latest" to use most recent by pipeline_info.json date

        The pipeline directory should contain:
            <study_id>/
                results.json  # Required
                raw_results.json  # Optional
                info.json  # Required
            pipeline_info.json  # Required for config selection
        """
        pipeline_dir = Path(pipeline_dir).resolve()
        if not pipeline_dir.exists():
            raise ValueError(f"Pipeline directory does not exist: {pipeline_dir}")

        # Get pipeline name from directory name
        pipeline_name = pipeline_dir.name

        # Get version directory
        if version == "latest":
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

            version = str(max(versions))
            version_dir = pipeline_dir / version
        else:
            version_dir = pipeline_dir / version
            if not version_dir.exists():
                raise ValueError(f"Version directory not found: {version_dir}")

        # Get config directory
        if config_hash == "latest":
            # Find most recent by pipeline_info.json date
            config_dirs = [d for d in version_dir.glob("*/") if d.is_dir()]
            if not config_dirs:
                raise ValueError(f"No config directories found in {version_dir}")

            latest_date = None
            latest_config = None

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
            config_dir = version_dir / config_hash
            if not config_dir.exists():
                raise ValueError(f"Config directory not found: {config_dir}")
            pipeline_info = config_dir / "pipeline_info.json"
            if not pipeline_info.exists():
                raise ValueError(f"No pipeline_info.json found in {config_dir}")

        # Create pipeline metadata
        pipeline = PipelineInfo(
            name=pipeline_name,
            version=version,
            config_hash=config_hash,
            pipeline_info=pipeline_info,
        )
        self.pipelines[pipeline_name] = {"info": pipeline, "results": {}}

        # Add results to studies
        for study_dir in config_dir.glob("*/"):
            if not study_dir.is_dir():
                continue

            study_id = study_dir.name
            study = self.data.get(study_id)
            if not study:
                logger.warning(f"Found results for unknown study: {study_id}")
                continue

            # Get paths to result files
            result_path = study_dir / "results.json"
            raw_path = study_dir / "raw_results.json"
            info_path = study_dir / "info.json"

            # Check required files exist
            if not result_path.exists():
                logger.warning(f"No results.json found for study {study_id}")
                continue

            if not info_path.exists():
                logger.warning(f"No info.json found for study {study_id}")
                continue

            # Create and add PipelineRunResult
            result = PipelineRunResult(
                pipeline=pipeline,
                result=result_path,
                info=info_path,
                raw_result=raw_path if raw_path.exists() else None,
            )
            study.add_pipeline_result(result)
            self.pipelines[pipeline_name]["results"][study_id] = result

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return an item from the dataset."""
        return self.data[idx]


class PipelineInputFilter:
    """Filter for pipeline inputs."""

    def __init__(self, pipeline, output_directory: Union[str, Path], overwrite=False):
        """Initialize the filter.

        Args:
            pipeline (Pipeline): The pipeline to filter.
            output_directory (Union[str, Path]): The output directory where the pipeline
                has been previously run.
            overwrite (bool): Whether to overwrite the existing output.
        """
        self.output_directory = (
            Path(output_directory).resolve()
            if isinstance(output_directory, str)
            else output_directory.resolve()
        )
        self.pipeline = pipeline
        self.overwrite = overwrite

    def filter(self, dataset):
        """Filter the dataset."""
        pass

    def load_outputs(self):
        """Load the outputs."""
        pass
