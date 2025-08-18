import json
import pytest
from ns_extract.pipelines.semantic_embeddings.model import GeneralAPIEmbeddingExtractor
from ns_extract.dataset import Dataset


@pytest.mark.vcr(record_mode="once", filter_headers=["authorization"])
@pytest.mark.parametrize("text_source", ["title", "abstract", "full_text"])
def test_GeneralAPIEmbeddingExtractor(sample_data, tmp_path, text_source, setup_tiktoken_cache):
    """Test the GeneralAPIEmbeddingExtractor pipeline."""
    extractor = GeneralAPIEmbeddingExtractor(
        extraction_model="text-embedding-3-small",
        env_variable="OPENAI_API_KEY",
        text_source=text_source,
    )
    dataset = Dataset(sample_data)
    output_dir = tmp_path / "semantic_embeddings"

    # Run extraction
    extractor.transform_dataset(dataset, output_dir)

    # Check output structure
    version_dir = next(output_dir.glob(f"GeneralAPIEmbeddingExtractor/{extractor._version}/*"))
    assert version_dir.exists()

    # Check pipeline info
    info_path = version_dir / "pipeline_info.json"
    pipeline_info = json.loads(info_path.read_text())
    assert pipeline_info["version"] == "1.0.0"

    # Check study outputs
    study_dirs = [x for x in version_dir.glob("*") if x.is_dir()]
    expected_len = len([d for d in sample_data.iterdir() if d.is_dir()])
    assert len(study_dirs) == expected_len

    for study_dir in study_dirs:
        results_file = study_dir / "results.json"
        info_file = study_dir / "info.json"
        assert results_file.exists()
        assert info_file.exists()

        # Validate results schema
        results = json.loads(results_file.read_text())
        assert "embedding" in results
        assert isinstance(results["embedding"], list)
        assert all(isinstance(x, float) for x in results["embedding"])
