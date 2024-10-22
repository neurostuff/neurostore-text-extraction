from datetime import datetime
import hashlib
import json
# from neurostore_text_extraction.pipelines.dataset import Dataset
# from neurostore_text_extraction.pipelines.pipeline import Node

class WordCountExtraction:
    """Extract word count from the documents.

    Extract word counts from the text data.
    """
    _version = "1.0.0"

    _pipeline_type = "independent"

    _hash_args = ['prefer']

    def __init__(self, prefer="pubget"):
        """Initialize the word count extraction pipeline."""
        self.prefer = prefer

    def run(self, dataset, output_dir):
        """Run the word count extraction pipeline.
        Output metadata:
            - version
            - date
            - source
        """

        # dataset hash
        dataset_str = "_".join(list(dataset.data.keys()))
        # argument hash
        arg_str = '_'.join([str(getattr(self, arg)) for arg in self._hash_args])

        # full hash
        hash_str = hashlib.shake_256(f"{dataset_str}_{arg_str}".encode()).hexdigest(6)

        # find the most recent restults across all directories
        existing_results = {} # main key is study id, contains dates, inputs, and hash where the file came from
        # extract the hash from the directory
        for d in output_dir.glob(f"{self._version}/**/*"):
            if d.is_dir() and d.name in set(dataset.data.keys()):
                if (d / "info.json").exists():
                    with open(d / "info.json", "r") as f:
                        info = json.load(f)
                    found_info = {"date": info["date"], "inputs": info["inputs"], "hash": hashlib.md5(str(info).encode()).hexdigest()}
                    if existing_results.get(d.name) is None or info["date"] > existing_results[d.name]["date"]:
                        existing_results[d.name] = found_info

        hash_outdir = (output_dir / self._version / hash_str)
        hash_outdir.mkdir(parents=True, exist_ok=True)

        for db_id, study in dataset.data.items():
            if self.prefer == "pubget" and getattr(study.pubget, "text", None) is not None:
                text_file = study.pubget.text
            elif self.prefer == "ace" and getattr(study.ace, "text", None) is not None:
                text_file = study.ace.text
            else:
                text_file = getattr(study.pubget, "text", None) or getattr(study.ace, "text", None)

            if text_file is None:
                raise ValueError(f"No text found for {db_id}")

            with open(text_file, "r") as f:
                text = f.read()

            text_len = len(text.split())

            # make the directory if it doesn't exist
            study_outdir = (hash_outdir / db_id)
            study_outdir.mkdir(parents=True, exist_ok=True)
            with open(study_outdir / "results.json", "w") as f:
                json.dump({"word_count": text_len}, f)

            with open(study_outdir / "info.json", "w") as f:
                json.dump({
                    "date": datetime.now().isoformat(),
                    "inputs": {
                        str(text_file): hashlib.md5(text.encode()).hexdigest()
                    }
                }, f
                )
