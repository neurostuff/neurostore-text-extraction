from ns_extract.dataset import Dataset


def test_dataset(sample_data):
    dataset = Dataset(sample_data)

    assert len(dataset) == 3
