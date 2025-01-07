import os
import csv
import pandas as pd
from video_processing import process_video


def save_metadata(video_path, metadata_csv, annotation_file=None):
    """
    Speichert Metadaten eines Videos in einer CSV-Datei und prüft optional die Schrittzählung mit Annotationen.

    Parameters:
        video_path (str): Pfad zur Videodatei.
        metadata_csv (str): Pfad zur zentralen Metadatendatei.
        annotation_file (str): Pfad zur Excel-Datei mit manuellen Schritt-Annotationen (optional).
    """
    # Initialize or load summary from a global variable
    if not hasattr(save_metadata, "summary"):
        save_metadata.summary = {"matches": 0, "non_matches": [], "no_annotations": []}

    summary = save_metadata.summary

    # Extrahiere Metadaten und Schritte
    result = process_video(video_path, display_video=False)
    if not result:
        print(f"Fehler beim Extrahieren der Metadaten für '{video_path}'.")
        return

    metadata, _, _, _, _ = result  # Ergebnisse von process_video entpacken
    required_keys = ["resolution", "fps", "duration_seconds", "creation_time", "num_steps"]
    if not all(key in metadata for key in required_keys):
        print(f"Unvollständige Metadaten für '{video_path}'. Überspringe.")
        return

    # Extrahiere die berechneten Schritte
    calculated_steps = metadata["num_steps"]

    # Prüfe, ob Annotationen existieren, falls ein Annotation-File angegeben ist
    if annotation_file:
        if not os.path.exists(annotation_file):
            print(f"Annotationsdatei '{annotation_file}' existiert nicht. Bitte erstellen Sie diese.")
            return

        # Lade die Annotationen
        annotations = pd.read_excel(annotation_file)

        # Basisdateiname des Videos extrahieren
        video_filename = os.path.basename(video_path)

        # Prüfe, ob Annotationen für das Video existieren
        if video_filename in annotations["filename"].values:
            # Hole die manuelle Schrittzählung
            manual_steps = annotations.loc[annotations["filename"] == video_filename, "manual_steps"].iloc[0]

            # Vergleiche berechnete Schritte mit manuellen Schritten
            if calculated_steps == manual_steps:
                print(f"Schrittzählungen stimmen überein für '{video_filename}' (Manuell: {manual_steps}, Berechnet: {calculated_steps}).")
                summary["matches"] += 1
            else:
                print(f"Schrittzählungen stimmen NICHT überein für '{video_filename}' (Manuell: {manual_steps}, Berechnet: {calculated_steps}). Überspringe.")
                summary["non_matches"].append({
                    "filename": video_filename,
                    "manual_steps": manual_steps,
                    "calculated_steps": calculated_steps
                })
                return  # Nicht speichern, wenn Schritte nicht übereinstimmen
        else:
            print(f"Keine Annotation gefunden für '{video_filename}'. Verarbeite mit berechneten Schritten.")

    # CSV-Datei erstellen, falls nicht vorhanden
    if not os.path.exists(metadata_csv):
        with open(metadata_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "resolution", "fps", "duration_seconds", "creation_time", "num_steps"])

    # Schreibe Metadaten in CSV
    if not is_video_in_csv(metadata_csv, os.path.basename(video_path)):
        with open(metadata_csv, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                os.path.basename(video_path),
                metadata["resolution"],
                metadata["fps"],
                metadata["duration_seconds"],
                metadata["creation_time"],
                metadata["num_steps"]
            ])
        print(f"Metadaten für '{video_path}' gespeichert in '{metadata_csv}'.")
    else:
        print(f"Metadaten für '{video_path}' sind bereits in '{metadata_csv}'. Überspringe.")


def is_video_in_csv(csv_file, video_filename):
    """
    Checks if a video's metadata is already in the CSV file.

    Parameters:
        csv_file (str): Path to the CSV file.
        video_filename (str): Name of the video file.

    Returns:
        bool: True if the video is already in the CSV, False otherwise.
    """
    if not os.path.exists(csv_file):
        return False

    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["filename"] == video_filename:
                return True
    return False


def print_summary():
    """
    Prints the summary of matching and non-matching annotations.

    This function should be called at the end of the main processing loop.
    """
    summary = getattr(save_metadata, "summary", {"matches": 0, "non_matches": []})

    print("\n=== Summary ===")
    print(f"Total Matches: {summary['matches']}")
    print(f"Total Non-Matches: {len(summary['non_matches'])}")

    if summary["non_matches"]:
        print("\nNon-Matching Annotations:")
        for item in summary["non_matches"]:
            print(f"  - {item['filename']}: Manual Steps = {item['manual_steps']}, Calculated Steps = {item['calculated_steps']}")