from pydub import AudioSegment
import os

def split_audio_into_parts(file_path, num_parts=5):
    try:
        # Lade die Audiodatei
        audio = AudioSegment.from_file(file_path)
        
        # Berechne die Länge jedes Teils
        part_length = len(audio) // num_parts
        
        # Teile die Audiodatei in num_parts Teile
        parts = [audio[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
        
        # Erstelle die Dateinamen für die Teile
        base_name, ext = os.path.splitext(file_path)
        
        for i, part in enumerate(parts):
            part_file = f"{base_name}_part{i+1}{ext}"
            part.export(part_file, format=ext[1:])  # Entferne das Punktzeichen vor der Erweiterung
            print(f"Teil {i+1} gespeichert als: {part_file}")
        
    except Exception as e:
        print(f"Fehler beim Verarbeiten der Datei {file_path}: {e}")

def process_folder(folder_path):
    # Gehe durch alle Dateien im Ordner
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Prüfe, ob es sich um eine Datei handelt
        if os.path.isfile(file_path):
            split_audio_into_parts(file_path, num_parts=5)

# Beispielverwendung
folder_path = "to_big_audio"
process_folder(folder_path)
