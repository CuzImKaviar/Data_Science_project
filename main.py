import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Extrahiere Merkmale aus einer Audiodatei
def extract_features(audio_file, n_mfcc=11, hop_length=1024, n_fft=4096):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    features = np.concatenate((mfccs, chroma), axis=0)
    return np.mean(features.T, axis=0)

# Lade die Audiodateien und extrahiere Merkmale
def load_data(folder_path, n_mfcc=11, hop_length=1024, n_fft=4096):
    features = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                try:
                    feature = extract_features(file_path, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    features.append(feature)
                    labels.append(label)
                except Exception as e:
                    print("Fehler beim Verarbeiten der Datei {}: {}".format(file_path, e))
    return np.array(features), np.array(labels)

# Trainiere den Klassifikator
def train_classifier(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features_scaled, labels)
    return clf, scaler

# Klassifiziere eine Audiodatei
def classify_audio(audio_file, classifier, scaler, label_encoder, n_mfcc=11, hop_length=1024, n_fft=4096):
    features = extract_features(audio_file, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    features_scaled = scaler.transform([features])  # Skalieren der Features
    predicted_label = classifier.predict(features_scaled)[0]
    predicted_class = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_class

# Funktion zum automatisierten Testen des Modells
def evaluate_model(classifier, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

# Streamlit-Anwendung
def main():
    st.title("Tierstimmen-Erkennung")
    folder_path = "Animal-SDataset" # Standardwert
    uploaded_file = st.file_uploader("Bitte lade eine Audiodatei hoch", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Erkenne Tierstimme"):
            try:
                predicted_class = classify_audio(uploaded_file, classifier, scaler, label_encoder, n_mfcc=11, hop_length=1024, n_fft=4096)
                st.write("Erkannte Tierstimme:", predicted_class)
            except Exception as e:
                st.error("Fehler beim Erkennen der Tierstimme: {}".format(e))

    # Trainiere und bewerte das Modell
    st.sidebar.subheader("Modelltraining und -bewertung")
    with st.spinner("Modell wird trainiert..."):
        features, labels = load_data(folder_path, n_mfcc=11, hop_length=1024, n_fft=4096)
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
        classifier, scaler = train_classifier(X_train, y_train)
        accuracy, report = evaluate_model(classifier, scaler, X_test, y_test)
        st.sidebar.write("Modellgenauigkeit:", accuracy)
        st.sidebar.text_area("Klassifikationsbericht:", report)

if __name__ == "__main__":
    main()