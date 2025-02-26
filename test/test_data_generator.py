import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.generator import compute_enmo

def test_compute_enmo():
    test_data = {"X": np.array([1, 0, -1]), "Y": np.array([0, 1, -1]), "Z": np.array([1, 1, 1])}
    expected_output = np.maximum(np.sqrt(test_data["X"]**2 + test_data["Y"]**2 + test_data["Z"]**2) - 1, 0)
    assert np.allclose(compute_enmo(test_data), expected_output)
