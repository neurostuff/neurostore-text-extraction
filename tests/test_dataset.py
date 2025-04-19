from ns_extract.dataset import Dataset


def test_dataset(sample_data):
    dataset = Dataset(sample_data)

    # Count actual directories in sample_data
    expected_len = len([d for d in sample_data.iterdir() if d.is_dir()])
    assert len(dataset) == expected_len
