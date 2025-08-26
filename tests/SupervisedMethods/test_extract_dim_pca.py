import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
import SupervisedMethods.extract_dim_pca as extract_dim_pca

def test_load_concept_dataset() -> None:
    """
    Test the load_concept_dataset function.
    """
    dataset_dict = extract_dim_pca.load_concept_dataset("pyvene/axbench-concept16k_v2")
    for key, value in dataset_dict.items():
        print(key, len(value))
    assert isinstance(dataset_dict, dict)
    assert len(dataset_dict) > 0
    for concept, data in dataset_dict.items():
        assert isinstance(concept, str)
        assert isinstance(data, list)
        assert len(data) > 10

if __name__ == "__main__":
    test_load_concept_dataset()