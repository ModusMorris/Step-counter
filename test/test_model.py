import torch
import numpy as np
from cnn.model_step_counter import StepCounterCNN

def test_model_forward_pass():
    model = StepCounterCNN(window_size=64)
    sample_input = torch.randn(1, 3, 64)  # Batch=1, Channels=3 (ENMO links & rechts), Window=64
    output = model(sample_input)
    assert output.shape == (1, 1)  # Modell sollte eine einzige Wahrscheinlichkeitsausgabe liefern.
