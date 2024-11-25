from pathlib import Path
from ns_pipelines import WordCountExtractor, WordDevianceExtractor
from ns_pipelines.dataset import Dataset


def test_WordCountExtractor(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    wce = WordCountExtractor()
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_count"
    wce.run(dataset, output_dir)
    # rerun where the output directory already exists
    # no ouputs generated
    wce.run(dataset, output_dir)
    # rerun with preference of ace
    wce_ace = WordCountExtractor(input_sources=("ace", "pubget"))
    wce_ace.run(dataset, output_dir)
    assert True


def test_WordDevianceExtractor(sample_data, tmp_path):
    """Test the word deviance extraction pipeline."""
    wde = WordDevianceExtractor()
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_deviance"
    wde.run(dataset, output_dir)
    # rerun where the output directory already exists
    # no ouputs generated
    wde.run(dataset, output_dir)
    # rerun with preference of ace
    wde_ace = WordDevianceExtractor(input_sources=("ace", "pubget"))
    wde_ace.run(dataset, output_dir)
    assert True
