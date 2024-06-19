import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st

def plot_waveform_and_spectrogram(y, sr):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title="Waveform")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set(title="Spectrogram")

    st.pyplot(fig)

def plot_mfccs(y, sr, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    st.pyplot(fig)

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    st.pyplot(fig)
