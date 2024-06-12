import os
import librosa
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from joblib import dump, load

folder_path = "Animal-SDataset"

# Extrahiere Merkmale aus einer Audiodatei
def extract_features(y, sr, n_mfcc=11, hop_length=1024, n_fft=4096):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        mfccs = np.mean(mfccs, axis=1)
        chroma = np.mean(chroma, axis=1)
        mel = np.mean(mel, axis=1)
        zcr = np.mean(zcr, axis=1)
        
        features = np.concatenate((mfccs, chroma, mel, zcr))
        return features
    except Exception as e:
        print(f"Fehler beim Extrahieren der Merkmale: {e}")
        return None

# Datenaugmentation durch Pitch-Shifting und Time-Stretching
def augment_audio(y, sr):
    try:
        pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        time_stretched = librosa.effects.time_stretch(y, rate=1.1)
        return [y, pitch_shifted, time_stretched]
    except Exception as e:
        print(f"Fehler bei der Datenaugmentation: {e}")
        return [y]

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
                    y, sr = librosa.load(file_path)
                    augmented_audios = augment_audio(y, sr)
                    for audio in augmented_audios:
                        feature = extract_features(audio, sr, n_mfcc, hop_length, n_fft)
                        if feature is not None:
                            features.append(feature)
                            labels.append(label)
                        else:
                            print(f"Feature-Extraktion fehlgeschlagen für {file_path}")
                except Exception as e:
                    print(f"Fehler beim Verarbeiten der Datei {file_path}: {e}")
    return np.array(features), np.array(labels)

# Trainiere den Klassifikator
def train_classifier(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    clf.fit(features_scaled, labels)
    return clf.best_estimator_, scaler

# Klassifiziere eine Audiodatei
def classify_audio(audio_file, classifier, scaler, label_encoder, n_mfcc=11, hop_length=1024, n_fft=4096):
    y, sr = librosa.load(audio_file)
    features = extract_features(y, sr, n_mfcc, hop_length, n_fft)
    features_scaled = scaler.transform([features])  # Skalieren der Features
    predicted_label = classifier.predict(features_scaled)[0]
    predicted_class = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_class

# Funktion zum automatisierten Testen des Modells
def evaluate_model(classifier, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    return accuracy, f1, precision, recall, report

# Laden des Modells und der Objekte
# Lade des Modells und der Objekte
# Lade des Modells und der Objekte
classifier = None  # Initialize classifier variable
accuracy = None  # Initialize accuracy variable

# Lade des Modells und der Objekte
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "random_forest_model.joblib")
scaler_path = os.path.join(model_dir, "scaler.joblib")
label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
X_test_path = os.path.join(model_dir, "X_test.npy")
y_test_path = os.path.join(model_dir, "y_test.npy")

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(label_encoder_path):
    classifier = load(model_path)
    scaler = load(scaler_path)
    label_encoder = load(label_encoder_path)
    
    # Try to load test data
    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        # Evaluate the model and get accuracy
        accuracy, _, _, _, _ = evaluate_model(classifier, scaler, X_test, y_test)
    except FileNotFoundError:
        print("Test data not found. Retraining the model might be necessary.")
else:
    # Train a new model if none exists
    features, labels = load_data(folder_path, n_mfcc=11, hop_length=1024, n_fft=4096)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    classifier, scaler = train_classifier(X_train, y_train)
    # Save the trained model and objects
    dump(classifier, model_path)
    dump(scaler, scaler_path)
    dump(label_encoder, label_encoder_path)
    np.save(X_test_path, X_test)
    np.save(y_test_path, y_test)
    # Evaluate the newly trained model and get accuracy
    accuracy, _, _, _, _ = evaluate_model(classifier, scaler, X_test, y_test)

# Streamlit-Anwendung
def main():
    global classifier  # Use the global classifier variable
    st.title("Tierstimmen-Erkennung")
    uploaded_file = st.file_uploader("Bitte lade eine Audiodatei hoch", type=["wav"], key="file_uploader")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Erkenne Tierstimme"):
            try:
                if classifier is None:
                    st.error("Das Modell wurde nicht geladen.")
                else:
                    predicted_class = classify_audio(uploaded_file, classifier, scaler, label_encoder, n_mfcc=11, hop_length=1024, n_fft=4096)
                    st.write("Erkannte Tierstimme:", predicted_class)
            except Exception as e:
                st.error("Fehler beim Erkennen der Tierstimme: {}".format(e))
    
    # Display accuracy if available
    if accuracy is not None:
        st.sidebar.markdown("**Modellgenauigkeit:**")
        st.sidebar.write(f"{accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()