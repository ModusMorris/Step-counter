import pandas as pd
from cnn.prediction import main

def test_full_pipeline(tmp_path):
    # Dummy CSV-Daten erstellen
    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    groundtruth_csv = tmp_path / "groundtruth.csv"

    pd.DataFrame({"X": [1,0,1], "Y": [0,1,0], "Z": [1,1,1]}).to_csv(left_csv, index=False)
    pd.DataFrame({"X": [1,1,0], "Y": [0,1,1], "Z": [1,0,1]}).to_csv(right_csv, index=False)
    pd.DataFrame({"Peaks": ["[1, 2]"]}).to_csv(groundtruth_csv, index=False)

    # Hauptfunktion testen (z. B. ob sie ohne Fehler läuft)
    main("dummy_model.pth", left_csv, right_csv, groundtruth_csv)