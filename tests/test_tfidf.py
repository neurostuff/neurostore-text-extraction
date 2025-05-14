import json
from ns_extract.pipelines import TFIDFExtractor
from ns_extract.dataset import Dataset


def test_TFIDFExtractor(sample_data, tmp_path):
    """Test the TFIDF extraction pipeline."""
    tfidf = TFIDFExtractor()
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "tfidf"

    # Initial run
    tfidf.transform_dataset(dataset, output_dir)

    # Verify directory structure and files
    version_dir = next(output_dir.glob("TFIDFExtractor/1.0.0/*"))
    assert version_dir.exists()

    # Check pipeline info
    info_path = version_dir / "pipeline_info.json"
    pipeline_info = json.loads(info_path.read_text())
    assert pipeline_info["version"] == "1.0.0"

    # Verify study outputs
    study_dirs = list([x for x in version_dir.glob("*") if x.is_dir()])
    expected_len = len([d for d in sample_data.iterdir() if d.is_dir()])
    assert len(study_dirs) == expected_len

    for study_dir in study_dirs:
        results_file = study_dir / "results.json"
        info_file = study_dir / "info.json"
        assert results_file.exists()
        assert info_file.exists()

        # Validate results schema
        results = json.loads(results_file.read_text())
        assert "terms" in results
        assert "tfidf_scores" in results
        assert isinstance(results["terms"], list)
        assert isinstance(results["tfidf_scores"], dict)
        assert len(results["terms"]) > 0
        assert len(results["tfidf_scores"]) > 0

        # Verify all terms have scores
        for term in results["terms"]:
            assert term in results["tfidf_scores"]
            assert isinstance(results["tfidf_scores"][term], float)
            assert results["tfidf_scores"][term] > 0


def test_text_type_options(sample_data, tmp_path):
    """Test different text_type options for TFIDF extraction."""
    dataset = Dataset(sample_data)
    base_dir = tmp_path / "text_types"

    # Create extractors with different text_type settings
    full_text = TFIDFExtractor(text_type="full_text")
    abstract = TFIDFExtractor(text_type="abstract")
    both = TFIDFExtractor(text_type="both")

    # Run each extractor
    full_text_dir = base_dir / "full_text"
    abstract_dir = base_dir / "abstract"
    both_dir = base_dir / "both"

    full_text.transform_dataset(dataset, full_text_dir)
    abstract.transform_dataset(dataset, abstract_dir)
    both.transform_dataset(dataset, both_dir)

    # Get first study results from each
    study_id = next(s.name for s in sample_data.iterdir() if s.is_dir())

    full_path = next(full_text_dir.glob("TFIDFExtractor/1.0.0/*"))
    abs_path = next(abstract_dir.glob("TFIDFExtractor/1.0.0/*"))
    both_path = next(both_dir.glob("TFIDFExtractor/1.0.0/*"))

    full_results = json.loads((full_path / study_id / "results.json").read_text())
    abs_results = json.loads((abs_path / study_id / "results.json").read_text())
    both_results = json.loads((both_path / study_id / "results.json").read_text())

    # Verify each has some terms and scores
    assert len(full_results["terms"]) > 0
    assert len(abs_results["terms"]) > 0
    assert len(both_results["terms"]) > 0


def test_custom_vocabulary(sample_data, tmp_path):
    """Test custom vocabulary functionality."""
    dataset = Dataset(sample_data)
    base_dir = tmp_path / "custom_vocab"

    # Test with vocabulary dictionary
    vocab_dict = {"brain": 0, "study": 1, "analysis": 2, "results": 3}
    dict_tfidf = TFIDFExtractor(vocabulary=vocab_dict)
    dict_dir = base_dir / "vocab_dict"
    dict_tfidf.transform_dataset(dataset, dict_dir)

    # Test with custom terms list
    terms_list = ["brain", "study", "analysis", "results"]
    list_tfidf = TFIDFExtractor(custom_terms=terms_list)
    list_dir = base_dir / "vocab_list"
    list_tfidf.transform_dataset(dataset, list_dir)

    # Get first study results from each
    study_id = next(s.name for s in sample_data.iterdir() if s.is_dir())

    dict_path = next(dict_dir.glob("TFIDFExtractor/1.0.0/*"))
    list_path = next(list_dir.glob("TFIDFExtractor/1.0.0/*"))

    dict_results = json.loads((dict_path / study_id / "results.json").read_text())
    list_results = json.loads((list_path / study_id / "results.json").read_text())

    # Verify results only contain terms from custom vocabulary
    dict_terms = set(dict_results["terms"])
    list_terms = set(list_results["terms"])

    assert dict_terms.issubset(set(vocab_dict.keys()))
    assert list_terms.issubset(set(terms_list))

    # Verify both methods produce same results
    assert dict_results == list_results


def test_parallel_processing(sample_data, tmp_path):
    """Test that parallel processing works correctly with TFIDFExtractor."""
    # Create extractor
    tfidf = TFIDFExtractor()
    dataset = Dataset(sample_data)

    # Run with different num_workers values
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    # Run in serial and parallel mode
    tfidf.transform_dataset(dataset, serial_dir, num_workers=1)
    tfidf.transform_dataset(dataset, parallel_dir, num_workers=4)

    # Get results from both runs
    serial_version_dir = next(serial_dir.glob("TFIDFExtractor/1.0.0/*"))
    parallel_version_dir = next(parallel_dir.glob("TFIDFExtractor/1.0.0/*"))

    # Verify same number of studies processed
    serial_studies = list(serial_version_dir.glob("*/results.json"))
    parallel_studies = list(parallel_version_dir.glob("*/results.json"))
    assert len(serial_studies) == len(parallel_studies)

    # Compare results
    for serial_result in serial_studies:
        # Find matching parallel result
        study_id = serial_result.parent.name
        parallel_result = parallel_version_dir / study_id / "results.json"

        assert parallel_result.exists()

        # Compare contents - should be identical since TFIDF is deterministic
        serial_data = json.loads(serial_result.read_text())
        parallel_data = json.loads(parallel_result.read_text())
        assert serial_data == parallel_data
