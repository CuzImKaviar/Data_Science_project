import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_waveform_and_spectrogram(y, sr):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")

    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")

    plt.tight_layout()
    plt.show()

def plot_mfccs(y, sr, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Lade eine Audiodatei und plotte die Wellenform und das Spektrogramm
y, sr = librosa.load("path/to/audio/file.wav")
plot_waveform_and_spectrogram(y, sr)
plot_mfccs(y, sr, n_mfcc=20)

# Beispiel für die Evaluierung des Modells und das Plotten der Konfusionsmatrix
y_test = np.load(X_test_path)
y_pred = classifier.predict(scaler.transform(X_test))
plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
