import time
import numpy as np
from cnn.prediction import detect_steps

def test_step_detection_performance():
    sample_data = np.random.rand(10000, 2)  # 10.000 Frames mit ENMO-Werten
    start_time = time.time()
    detect_steps(None, "cpu", sample_data)  # Dummy-Aufruf ohne Modell
    assert time.time() - start_time < 5  # Sollte nicht länger als 5 Sekunden dauern.
