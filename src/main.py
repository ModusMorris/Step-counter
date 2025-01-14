import argparse
import os
from metadata_handler import save_metadata, print_summary
from generator import video_file_generator
from video_processing import process_video, visualize_data, save_step_data_to_csv


def process_and_visualize_video(video_path, output_root, metadata_csv, annotation_file=None):
    """
    Verarbeitet ein einzelnes Video, speichert Metadaten und Ergebnisse (PDF, CSV) in einem benannten Ordner.

    Parameters:
        video_path (str): Pfad zum Video.
        output_root (str): Root-Verzeichnis, in dem der Videoordner erstellt wird.
        metadata_csv (str): Pfad zur zentralen Metadatendatei.
        annotation_file (str): Pfad zur Excel-Datei mit manuellen Annotationen (optional).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_root, video_name)
    os.makedirs(video_folder, exist_ok=True)

    print(f"Verarbeite Video: {video_path}")

    # Speichern der Metadaten
    #save_metadata(video_path, metadata_csv, annotation_file if annotation_file else None)

    # Verarbeitung und Visualisierung
    result = process_video(video_path, display_video=False)
    if not result:
        print(f"Fehler beim Verarbeiten von {video_path}")
        return

    # Ergebnisse entpacken und saven
    metadata, joints_data, smoothed_data, peaks_data, step_counts_joint = result

    save_metadata(metadata, video_path, metadata_csv, annotation_file if annotation_file else None)

    # PDF erstellen und speichern
    pdf_path = os.path.join(video_folder, f"{video_name}.pdf")
    visualize_data(joints_data, smoothed_data, peaks_data, output_path=pdf_path)

    # CSV-Dateien speichern
    save_step_data_to_csv(video_folder, joints_data, smoothed_data, peaks_data, step_counts_joint)

    print(f"Verarbeitung abgeschlossen. Ergebnisse im Ordner: {video_folder}")


def process_all_videos_in_directory(root_dir, output_root, annotation_file=None):
    """
    Durchsucht ein Verzeichnis nach Videos und verarbeitet alle, die noch nicht verarbeitet wurden.
    Speichert Metadaten und Ergebnisse.

    Parameters:
        root_dir (str): Verzeichnis mit Videos.
        output_root (str): Zielverzeichnis für alle Ergebnisse.
        annotation_file (str): Pfad zur Excel-Datei mit manuellen Annotationen (optional).
    """
    metadata_csv = os.path.join(output_root, "metadata.csv")
    for video_path in video_file_generator(root_dir, metadata_csv):
        process_and_visualize_video(video_path, output_root, metadata_csv, annotation_file)
    print("Alle Videos verarbeitet.")


def main():
    """
    Hauptfunktion für die Videoverarbeitung.
    """
    parser = argparse.ArgumentParser(description="Video Processing Script")
    parser.add_argument("--action", type=str, choices=["save_metadata", "process_data"], required=True,
                        help="Aktion: 'save_metadata' oder 'process_data'.")
    parser.add_argument("--root_dir", type=str,
                        help="Verzeichnis mit Videos (benötigt für 'save_metadata' und 'process_data').")
    parser.add_argument("--video_path", type=str,
                        help="Pfad zu einem bestimmten Video (nur für 'process_data').")
    parser.add_argument("--output_root", type=str, default="output",
                        help="Root-Verzeichnis für die Ausgabedateien.")
    parser.add_argument("--annotation_file", type=str, required=False,
                        help="Pfad zur Excel-Datei mit manuellen Annotationen.")

    args = parser.parse_args()

    if args.action == "process_data":
        if args.video_path:
            process_and_visualize_video(args.video_path, args.output_root, 
                                        os.path.join(args.output_root, "metadata.csv"), 
                                        args.annotation_file)
        elif args.root_dir:
            process_all_videos_in_directory(args.root_dir, args.output_root, args.annotation_file)
        else:
            print("Fehler: Bitte entweder '--video_path' oder '--root_dir' angeben.")
    elif args.action == "save_metadata":
        if not args.root_dir:
            print("Fehler: '--root_dir' ist erforderlich für 'save_metadata'.")
            return

        metadata_csv = os.path.join(args.output_root, "metadata.csv")
        annotation_file = args.annotation_file or os.path.join(args.root_dir, "step_annotations.xlsx")
        for video_path in video_file_generator(args.root_dir, metadata_csv):
            save_metadata(video_path, metadata_csv, annotation_file)
        print_summary()
    else:
        print("Ungültige Aktion. Bitte 'save_metadata' oder 'process_data' wählen.")


if __name__ == "__main__":
    main()
