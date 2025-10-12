"""Test controller implementations."""
import pytest
import torch
from src.controller.rnn_controller import RNNController
from src.search_space.cnn_space import CNNSearchSpace

def test_rnn_controller():
    search_space = CNNSearchSpace(max_layers=3)
    controller = RNNController(search_space, hidden_size=64)

    architectures, log_probs, entropies = controller(batch_size=2)

    assert len(architectures) == 2
    assert log_probs.shape[0] == 2
    assert entropies.shape[0] == 2
