"""Pytest configuration."""
import pytest
import numpy as np

@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)
