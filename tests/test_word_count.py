from pathlib import Path
from pipelines.word_count.run import WordCountExtraction
from pipelines.dataset import Dataset


def test_word_count_extraction(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    wce = WordCountExtraction(prefer="pubget")
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_count"
    wce.run(dataset, output_dir)
    assert True


def test_word_count_extraction_ace(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    wce = WordCountExtraction(prefer="ace")
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_count"
    wce.run(dataset, output_dir)
    assert True
