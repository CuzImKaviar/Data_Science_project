from pydub import AudioSegment
import os

def split_audio(file_path):
    # Lade die Audiodatei
    audio = AudioSegment.from_file(file_path)
    
    # Berechne die Mitte der Audiodatei
    mid_point = len(audio) // 2
    
    # Teile die Audiodatei in zwei Hälften
    first_half = audio[:mid_point]
    second_half = audio[mid_point:]
    
    # Erstelle die Dateinamen für die beiden Hälften
    base_name, ext = os.path.splitext(file_path)
    first_half_file = f"{base_name}_first_half{ext}"
    second_half_file = f"{base_name}_second_half{ext}"
    
    # Speichere die beiden Hälften
    first_half.export(first_half_file, format=ext[1:])  # Entferne das Punktzeichen vor der Erweiterung
    second_half.export(second_half_file, format=ext[1:])
    
    print(f"Erste Hälfte gespeichert als: {first_half_file}")
    print(f"Zweite Hälfte gespeichert als: {second_half_file}")

def process_folder(folder_path):
    # Gehe durch alle Dateien im Ordner
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Prüfe, ob es sich um eine Datei handelt
        if os.path.isfile(file_path):
            try:
                split_audio(file_path)
            except Exception as e:
                print(f"Fehler beim Verarbeiten der Datei {file_path}: {e}")

# Beispielverwendung
folder_path = "to_big_audio"
process_folder(folder_path)
