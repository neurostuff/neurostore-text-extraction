from pathlib import Path
import json
from ns_extract.pipelines import WordCountExtractor, WordDevianceExtractor
from ns_extract.dataset import Dataset


def test_WordCountExtractor(sample_data, tmp_path):
    """Test the word count extraction pipeline."""
    wce = WordCountExtractor()
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_count"
    
    # Initial run
    wce.transform_dataset(dataset, output_dir)

    # Verify directory structure and files
    version_dir = next(output_dir.glob("WordCountExtractor/1.0.0/*"))
    assert version_dir.exists()
    
    # Check pipeline info
    pipeline_info = json.loads((version_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == "1.0.0"
    assert pipeline_info["type"] == "independent"
    
    # Verify study outputs
    # Glob for dirs
    study_dirs = list([x for x in version_dir.glob("*") if x.is_dir()])
    assert len(study_dirs) == 3
    for study_dir in study_dirs:
        results_file = study_dir / "results.json"
        info_file = study_dir / "info.json"
        assert results_file.exists()
        assert info_file.exists()
        
        # Validate results schema
        results = json.loads(results_file.read_text())
        assert "word_count" in results
        assert isinstance(results["word_count"], int)
            
    # Rerun - no changes, no new outputs
    initial_hash_dir = version_dir
    wce.transform_dataset(dataset, output_dir)
    assert len(list(output_dir.glob("WordCountExtractor/1.0.0/*"))) == 1
    
    # Test input source preference
    wce_ace = WordCountExtractor(input_sources=("ace", "pubget"))
    wce_ace.transform_dataset(dataset, output_dir)
    # Should create new hash dir due to different arguments
    assert len(list(output_dir.glob("WordCountExtractor/1.0.0/*"))) == 2


def test_parallel_processing(sample_data, tmp_path):
    """Test that parallel processing works correctly with WordCountExtractor."""
    # Create extractor
    wce = WordCountExtractor()
    dataset = Dataset(sample_data)
    
    # Run with different num_workers values
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    
    # Run in serial and parallel mode
    wce.transform_dataset(dataset, serial_dir, num_workers=1)
    wce.transform_dataset(dataset, parallel_dir, num_workers=4)

    # Get results from both runs
    serial_version_dir = next(serial_dir.glob("WordCountExtractor/1.0.0/*"))
    parallel_version_dir = next(parallel_dir.glob("WordCountExtractor/1.0.0/*"))

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
        
        # Compare contents
        serial_data = json.loads(serial_result.read_text())
        parallel_data = json.loads(parallel_result.read_text())
        assert serial_data == parallel_data

def test_WordDevianceExtractor(sample_data, tmp_path):
    """Test the word deviance extraction pipeline."""
    wde = WordDevianceExtractor()
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "word_deviance"
    
    # Initial run
    wde.transform_dataset(dataset, output_dir)
    
    # Verify directory structure and files
    version_dir = next(output_dir.glob("WordDevianceExtractor/1.0.0/*"))
    assert version_dir.exists()
    
    # Check pipeline info
    pipeline_info = json.loads((version_dir / "pipeline_info.json").read_text())
    assert pipeline_info["version"] == "1.0.0"
    assert pipeline_info["type"] == "dependent"
    
    # Verify study outputs
    study_dirs = list([x for x in version_dir.glob("*") if x.is_dir()])
    assert len(study_dirs) == 3
    for study_dir in study_dirs:
        results_file = study_dir / "results.json"
        info_file = study_dir / "info.json"
        assert results_file.exists()
        assert info_file.exists()
        
        # Validate results schema
        results = json.loads(results_file.read_text())
        assert "word_deviance" in results
        assert isinstance(results["word_deviance"], int)
    
    # Rerun - no changes, no new outputs
    wde.transform_dataset(dataset, output_dir)
    assert len(list(output_dir.glob("WordDevianceExtractor/1.0.0/*"))) == 1
    
    # Test input source preference
    wde_ace = WordDevianceExtractor(input_sources=("ace", "pubget"))
    wde_ace.transform_dataset(dataset, output_dir)
    # Should create new hash dir due to different arguments
    assert len(list(output_dir.glob("WordDevianceExtractor/1.0.0/*"))) == 2
