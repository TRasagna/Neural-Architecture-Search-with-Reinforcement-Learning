"""Test search spaces."""
import pytest
from src.search_space.cnn_space import CNNSearchSpace

def test_cnn_search_space():
    space = CNNSearchSpace(max_layers=5)

    # Test random sampling
    arch = space.sample_random_architecture()
    assert 'layers' in arch
    assert len(arch['layers']) <= 5

    # Test validation
    assert space.validate_architecture(arch)
