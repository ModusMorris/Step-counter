import time
import numpy as np
from cnn.prediction import detect_steps

def test_step_detection_performance():
    sample_data = np.random.rand(10000, 2)  # 10.000 frames with ENMO-values
    start_time = time.time()
    detect_steps(None, "cpu", sample_data)  # dummy call without model
    assert time.time() - start_time < 5  # shouldn't last longer than 5 seconds
