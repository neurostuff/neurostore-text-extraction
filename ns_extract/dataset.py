"""Dataset creation for processing inputs."""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re
import json
from typing import Union, Optional
import logging

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
            self.tables = {f'{t:03}': {"metadata": None, "contents": None} for t in range(num_tables)}

            for tf in tables_files:
                table_number = tf.stem.split("_")[1]
                if tf.suffix == ".json":
                    key = "metadata"
                else:
                    key = "contents"
                self.tables[table_number][key] = tf

@dataclass
class ProcessedData:
    coordinates: Path = None
    text: Path = None
    metadata: Path = None
    raw: Optional[Union['PubgetRaw', 'AceRaw']] = field(default=None)

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

    def __post_init__(self):
        self.dbid = self.study_dir.name

        # Load identifiers
        with open((self.study_dir / "identifiers.json"), "r") as ident_fp:
            ids = json.load(ident_fp)

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
            pubget_raw = PubgetRaw(
                xml=pubget_xml_path,
                tables_xml=tables_xml_path
            )

        # Load processed data
        for t in ["ace", "pubget"]:
            processed_dir = self.study_dir / "processed" / t
            if processed_dir.exists():
                try:
                    processed = ProcessedData(
                        coordinates=processed_dir / "coordinates.csv",
                        text=processed_dir / "text.txt",
                        metadata=processed_dir / "metadata.json",
                        raw = ace_raw if t == "ace" else pubget_raw
                    )
                except ValueError as e:
                    logger.error(f"Error loading processed data for {self.dbid}: {e}")
                    continue

                setattr(self, t, processed)


class Dataset:
    """Dataset class for processing inputs."""

    def __init__(self, input_directory):
        """Initialize the dataset."""
        self.data = self.load_directory(input_directory)

    def slice(self, ids):
        """Slice the dataset."""
        deepcopy_obj = deepcopy(self)
        deepcopy_obj.data = {k: v for k, v in deepcopy_obj.data.items() if k in ids}
        return deepcopy_obj

    def load_directory(self, input_directory):
        """Load the input directory."""
        if isinstance(input_directory, str):
            input_directory = Path(input_directory)
        
        if not input_directory.exists():
            raise ValueError(
                f"Input directory {input_directory} does not exist.")
            
        pattern = re.compile(r'^[a-zA-Z0-9]{12}$')
        sub_directories = input_directory.glob("[0-9A-Za-z]*")
        study_directories = [
            dir_ for dir_ in sub_directories
            if dir_.is_dir() and pattern.match(dir_.name)
        ]

        dset_data = {}

        for study_dir in study_directories:
            study_obj = Study(study_dir=study_dir)

            dset_data[study_obj.dbid] = study_obj

        return dset_data
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
            output_directory (Union[str, Path]): The output directory where the pipeline has been previously run.
            overwrite (bool): Whether to overwrite the existing output.
        """
        self.output_directory = Path(output_directory) if isinstance(output_directory, str) else output_directory
        self.pipeline = pipeline
        self.overwrite = overwrite

    def filter(self, dataset):
        """Filter the dataset."""
        pass

    def load_outputs(self):
        """Load the outputs."""
        pass
