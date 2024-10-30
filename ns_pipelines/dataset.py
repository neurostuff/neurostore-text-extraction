"""Dataset creation for processing inputs."""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re
import json
from typing import Union, Optional

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

@dataclass
class PubgetRaw:
    xml: Path
    tables: dict = None
    tables_xml: Path = None

@dataclass
class ProcessedData:
    coordinates: Path = None
    text: Path = None
    metadata: Path = None
    raw: Optional[Union['PubgetRaw', 'AceRaw']] = field(default=None)

@dataclass
class Study:
    dbid: str
    doi: str = None
    pmid: str = None
    pmcid: str = None
    ace: ProcessedData = field(default_factory=ProcessedData)
    pubget: ProcessedData = field(default_factory=ProcessedData)


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
        """Load the input directory.
        input_directory (str): The input directory containing the text.
        processed (bool): Whether the input text is already processed.
        source (str): The source of the input text.
                      (ace or pubget, if None, tries to find both)
        """
        pattern = re.compile(r'^[a-zA-Z0-9]{12}$')

        sub_directories = input_directory.glob("[0-9A-Za-z]*")

        study_directories = [
            dir_ for dir_ in sub_directories
            if dir_.is_dir() and pattern.match(dir_.name)
        ]

        dset_data = {}

        for study_dir in study_directories:

            study_id = study_dir.name
            study_obj = Study(dbid=study_id)
            # associate IDs with study object
            with open((study_dir / "identifiers.json"), "r") as ident_fp:
                ids = json.load(ident_fp)

            study_obj.doi = ids["doi"] or None
            study_obj.pmid = ids["pmid"] or None
            study_obj.pmcid = ids["pmcid"] or None

            source_dir = study_dir / "source"

            # check if the source ace directory exists and load appropriate files
            if (source_dir / "ace").exists():
                study_obj.ace.raw = AceRaw(html=source_dir / "ace" / f"{study_obj.pmid}.html")

            # check if the source pubget directory exists and load appropriate files
            if (source_dir / "pubget").exists():
                study_obj.pubget.raw = PubgetRaw(
                    xml=source_dir / "pubget" / f"{study_obj.pmcid}.xml",
                )
                study_obj.pubget.raw.tables_xml = source_dir / "pubget" / "tables" / "tables.xml"

                tables_files = (source_dir / "pubget" / "tables").glob("*.xml")
                tables_files = [t for t in tables_files if t.name != "tables.xml"]

                num_tables = len(tables_files) // 2
                study_obj.pubget.raw.tables = {
                    '{0:03}'.format(t): {"metadata": None, "contents": None}
                    for t in range(num_tables)
                }

                for tf in tables_files:
                    table_number = tf.stem.split("_")[1]
                    if tf.suffix == ".json":
                        key = "metadata"
                    else:
                        key = "contents"

                    study_obj.pubget.raw.tables[table_number][key] = tf

            # processed directory
            processed_dir = study_dir / "processed"
            if (processed_dir / "ace").exists():
                study_obj.ace.coordinates = processed_dir / "ace" / "coordinates.csv"
                study_obj.ace.text = processed_dir / "ace" / "text.txt"
                study_obj.ace.metadata = processed_dir / "ace" / "metadata.json"

            if (processed_dir / "pubget").exists():
                study_obj.pubget.coordinates = processed_dir / "pubget" / "coordinates.csv"
                study_obj.pubget.text = processed_dir / "pubget" / "text.txt"
                study_obj.pubget.metadata = processed_dir / "pubget" / "metadata.json"

            dset_data[study_id] = study_obj

        return dset_data

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return an item from the dataset."""
        return self.data[idx]



class PipelineInputFilter:
    """Filter for pipeline inputs."""

    def __init__(self, pipeline, output_directory, overwrite=False):
        """Initialize the filter.

        pipeline (Pipeline): The pipeline to filter.
        output_directory (str): The output directory where the pipeline has been previously run.
        overwrite (bool): Whether to overwrite the existing output
        """

    def filter(self, dataset):
        """Filter the dataset."""
        pass

    def load_outputs(self):
        """Load the outputs."""
        pass
