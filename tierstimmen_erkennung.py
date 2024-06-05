import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

# Pfad zu den Audiodateien und zum gespeicherten Modell
data_dir = 'Animal-SDataset'
model_path = 'best_model.pkl'

# Funktion zur Datenaufbereitung
def prepare_data(data_dir):
    audio_data = []
    labels = []
    sample_rates = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path)
                label = os.path.basename(root)  # Ordnername ist das Label
                audio_data.append(y)
                labels.append(label)
                sample_rates.append(sr)
    return audio_data, labels, sample_rates

# Funktion zur Segmentierung und Normalisierung
def segment_audio(y, sr, segment_duration=2):
    segments = []
    segment_length = int(segment_duration * sr)
    for start in range(0, len(y), segment_length):
        end = start + segment_length
        if end <= len(y):
            segments.append(y[start:end])
    return segments

# Funktion zur Feature-Extraktion
def extract_features(segment, sr):
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Streamlit-Anwendung
st.title('Tierstimmen-Erkennung')

# Training des Modells, falls es noch nicht existiert
if not os.path.exists(model_path):
    st.write("Trainiere das Modell...")
    
    # Datenaufbereitung
    audio_data, labels, sample_rates = prepare_data(data_dir)
    
    st.write(f"Anzahl geladener Audiodateien: {len(audio_data)}")
    
    # Normalisiere und segmentiere Audiodaten
    audio_data_normalized = [librosa.util.normalize(y) for y in audio_data]
    audio_segments = [segment for y, sr in zip(audio_data_normalized, sample_rates) for segment in segment_audio(y, sr)]
    features = [extract_features(segment, sr) for segment, sr in zip(audio_segments, sample_rates)]
    
    st.write(f"Anzahl extrahierter Segmente: {len(audio_segments)}")

    # Datenaufteilung
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Hyperparameter-Tuning
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Modell speichern
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Evaluierung
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(classification_report(y_test, y_pred))

    accuracy = report['accuracy']
    st.write(f'Genauigkeit: {accuracy}')
else:
    st.write("Lade das trainierte Modell...")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
    st.success("Das trainierte Modell wurde erfolgreich geladen.")

# Funktion zur Vorhersage
def predict_audio(audio_file):
    y, sr = librosa.load(audio_file)
    segment = librosa.util.normalize(y)
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return best_model.predict([mfccs_mean])

# Audiodatei hochladen und vorhersagen
uploaded_file = st.file_uploader('Lade eine Audiodatei hoch', type=['wav', 'mp3'])
if uploaded_file is not None:
    st.write("Datei hochgeladen:", uploaded_file.name)
    prediction = predict_audio(uploaded_file)
    st.write(f'Die erkannte Tierstimme ist: {prediction[0]}')
else:
    st.write("Bitte eine Audiodatei hochladen")
