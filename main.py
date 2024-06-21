import os
import librosa
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from joblib import dump, load
import soundfile as sf
import sounddevice as sd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = "Animal-SDataset"

def record_audio(filename='recorded_audio.wav', duration=5, samplerate_=48000):
    try:
        print("Bitte sprechen Sie jetzt...")
        myrecording = sd.rec(int(duration * samplerate_), samplerate=samplerate_, channels=2)
        sd.wait()
        sf.write(filename, myrecording, samplerate_)
        print(f"Aufnahme gespeichert als {filename}")
        return filename
    except Exception as e:
        print(f"Fehler bei der Aufnahme: {e}")
        return None

def preprocess_audio(y, sr):
    try:
        y = librosa.effects.preemphasis(y)
        y = librosa.effects.remix(y, intervals=librosa.effects.split(y, top_db=20))
        return y
    except Exception as e:
        print(f"Fehler bei der Vorverarbeitung: {e}")
        return y

def extract_features(y, sr, n_mfcc=25, hop_length=1024, n_fft=4096):
    try:
        y = preprocess_audio(y, sr)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        mfccs = np.mean(mfccs, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        chroma = np.mean(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        mel = np.mean(mel, axis=1)

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr = np.mean(zcr, axis=1)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_bands=6, fmin=20.0)
        spectral_contrast = np.mean(spectral_contrast, axis=1)

        features = np.concatenate((mfccs, chroma, mel, zcr, spectral_contrast))
        return features
    except Exception as e:
        print(f"Fehler beim Extrahieren der Merkmale: {e}")
        return None

def augment_audio(y, sr):
    augmented_audios = [y]
    try:
        augmented_audios.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
        augmented_audios.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))

        # Zeitstreckung
        augmented_audios.append(librosa.effects.time_stretch(y, rate=1.1))
        augmented_audios.append(librosa.effects.time_stretch(y, rate=0.9))

        # Geschwindigkeitsänderung (Speed Perturbation)
        augmented_audios.append(librosa.effects.time_stretch(y, rate=0.95))
        augmented_audios.append(librosa.effects.time_stretch(y, rate=1.05))

        # Rauschbehaftung
        augmented_audios.append(y + 0.005 * np.random.randn(len(y)))

        # Änderungen der Raumklangcharakteristik (z.B. Hall/Echo)
        # Hier ein Beispiel mit Pre-Emphasis und Trimmen
        augmented_audios.append(librosa.effects.preemphasis(y))
        augmented_audios.append(librosa.effects.trim(y)[0])

        # Veränderungen der Dynamik
        augmented_audios.append(y * (1 + 0.2 * np.random.randn(len(y))))

        # Sie können auch Kombinationen von verschiedenen Techniken verwenden
        # Beispiel: Kombination von Tonhöhenverschiebung, Zeitstreckung und Rauschbehaftung
        augmented_audios.append(librosa.effects.pitch_shift(librosa.effects.time_stretch(y, rate=1.1), sr=sr, n_steps=1))
        augmented_audios.append(librosa.effects.pitch_shift(librosa.effects.time_stretch(y, rate=0.9), sr=sr, n_steps=-1) + 0.005 * np.random.randn(len(y)))

        # Augmentierte Daten sind nun in der Liste augmented_audios gespeichert
    except Exception as e:
        print(f"Fehler bei der Datenaugmentation: {e}")
    return augmented_audios

def load_data(folder_path, n_mfcc=20, hop_length=1024, n_fft=4096):
    features = []
    labels = []
    file_list = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith(".wav") or file.endswith(".ogg")]
    progress_bar = st.progress(0)
    
    for i, file_path in enumerate(tqdm(file_list, desc="Daten werden geladen")):
        label = os.path.basename(os.path.dirname(file_path))
        try:
            y, sr = librosa.load(file_path, sr=None)
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
        progress_bar.progress((i + 1) / len(file_list))
    
    progress_bar.empty()
    return np.array(features), np.array(labels)

def train_classifier(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 15, 20, 25],
        'min_samples_split': [3, 4, 5, 6, 7]
    }
    
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, verbose=10, n_jobs=-1)
    
    progress_bar = st.progress(0)
    with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * 3, desc="Model training") as pbar:
        for i in range(3):  # 3-fold cross-validation
            clf.fit(features_scaled, labels)
            pbar.update(1)
            progress_bar.progress((i + 1) / 3)
    
    progress_bar.empty()
    print("Beste Parameterkombination: ", clf.best_params_)
    print("Beste Genauigkeit: ", clf.best_score_)
    
    return clf.best_estimator_, scaler

def classify_audio(audio_file, classifier, scaler, label_encoder, n_mfcc=11, hop_length=1024, n_fft=4096):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        features = extract_features(y, sr, n_mfcc, hop_length, n_fft)
        features_scaled = scaler.transform([features])
        predicted_label = classifier.predict(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([predicted_label])[0]
        return predicted_class
    except Exception as e:
        print(f"Fehler bei der Klassifizierung der Audiodatei {audio_file}: {e}")
        return None

def evaluate_model(classifier, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    return accuracy, f1, precision, recall, report

def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

classifier = None
accuracy = None

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
    
    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        accuracy, _, _, _, _ = evaluate_model(classifier, scaler, X_test, y_test)
    except FileNotFoundError:
        print("Testdaten nicht gefunden. Das Modell muss möglicherweise neu trainiert werden.")
else:
    st.write("Lade und verarbeite Daten...")
    features, labels = load_data(folder_path, n_mfcc=11, hop_length=1024, n_fft=4096)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    
    st.write("Trainiere das Modell...")
    classifier, scaler = train_classifier(X_train, y_train)
    
    dump(classifier, model_path)
    dump(scaler, scaler_path)
    dump(label_encoder, label_encoder_path)
    np.save(X_test_path, X_test)
    np.save(y_test_path, y_test)
    accuracy, _, _, _, _ = evaluate_model(classifier, scaler, X_test, y_test)

def main():
    global classifier
    if 'audio_recorded' not in st.session_state:
        st.session_state.audio_recorded = False
        st.session_state.audio_filename = None

    st.title("Tierstimmen-Erkennung")
    st.write("Wählen Sie eine der folgenden Optionen:")

    option = st.selectbox('', ('Audio hochladen', 'Audio aufnehmen'))

    if option == 'Audio hochladen':
        uploaded_file = st.file_uploader("Bitte lade eine Audiodatei hoch", type=["wav", "ogg"], key="file_uploader")
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav' if uploaded_file.name.endswith('.wav') else 'audio/ogg')
            if st.button("Erkenne Tierstimme"):
                try:
                    if classifier is None:
                        st.error("Das Modell wurde nicht geladen.")
                    else:
                        predicted_class = classify_audio(uploaded_file, classifier, scaler, label_encoder, n_mfcc=11, hop_length=1024, n_fft=4096)
                        if predicted_class is not None:
                            st.write("Erkannte Tierstimme:", predicted_class)
                        else:
                            st.error("Fehler beim Erkennen der Tierstimme.")
                except Exception as e:
                    st.error("Fehler beim Erkennen der Tierstimme: {}".format(e))

    elif option == 'Audio aufnehmen':
        duration = st.slider("Dauer der Aufnahme (in Sekunden)", min_value=1, max_value=10, value=5)

        samplerate_mic_ = st.number_input("Samplerate des Mikrofons in kHz", min_value=8.0, max_value=48.0, value=48.0, step=0.1)
        samplerate_mic = int(samplerate_mic_ * 1000)

        if st.button("Aufnahme starten"):
            audio_filename = record_audio(duration=duration, samplerate_=samplerate_mic)
            if audio_filename:
                st.session_state.audio_recorded = True
                st.session_state.audio_filename = audio_filename
                st.success("Aufnahme erfolgreich!")
                st.audio(audio_filename, format='audio/wav')

        if st.session_state.audio_recorded:
            if st.button("Erkenne Tierstimme"):
                try:
                    predicted_class = classify_audio(st.session_state.audio_filename, classifier, scaler, label_encoder, n_mfcc=11, hop_length=1024, n_fft=4096)
                    if predicted_class is not None:
                        st.write("Erkannte Tierstimme:", predicted_class)
                    else:
                        st.error("Fehler beim Erkennen der Tierstimme.")
                except Exception as e:
                    st.error("Fehler beim Erkennen der Tierstimme: {}".format(e))

    if accuracy is not None:
        st.sidebar.markdown("**Modellgenauigkeit:**")
        st.sidebar.write(f"{accuracy * 100:.2f}%")
        
        if st.sidebar.button("Confusion Matrix anzeigen"):
            y_pred = classifier.predict(scaler.transform(X_test))
            plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)

if __name__ == "__main__":
    main()
