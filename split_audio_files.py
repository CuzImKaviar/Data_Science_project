from pydub import AudioSegment
import os


NUM_PARTS = 2

def split_audio_into_parts(file_path, num_parts):
    try:
        # Lade Audiodatei
        audio = AudioSegment.from_file(file_path)
        

        part_length = len(audio) // num_parts
        
        parts = [audio[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
        
        base_name, ext = os.path.splitext(file_path)
        
        for i, part in enumerate(parts):
            part_file = f"{base_name}_part{i+1}{ext}"
            part.export(part_file, format=ext[1:]) 
            print(f"Teil {i+1} gespeichert als: {part_file}")
        
        # Löschen der ursprünglichen Datei
        os.remove(file_path)
        print(f"Ursprüngliche Datei gelöscht: {file_path}")
        
    except Exception as e:
        print(f"Fehler beim Verarbeiten der Datei {file_path}: {e}")

def process_folder(folder_path, num_parts):

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        

        if os.path.isfile(file_path):
            split_audio_into_parts(file_path, num_parts)

folder_path = "to_big_audio"
process_folder(folder_path, NUM_PARTS)
