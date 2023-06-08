import streamlit as st
import parselmouth
from parselmouth.praat import call
import librosa.display
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from IPython.display import Audio

st.set_option('deprecation.showPyplotGlobalUse', False)

def generate_voice_report(audio_file):
    sound = parselmouth.Sound(audio_file)
    pitch = call(sound, "To Pitch", 0.0, 20, 100)
    pulse = call([sound, pitch], "To PointProcess (cc)")

    voice_report = parselmouth.praat.call([sound, pitch, pulse], "Voice report", 0.0, sound.get_total_duration(), 75,
                                          600, 1.3, 1.6, 0.03, 0.45).split("\n")
    return voice_report


def plot_waveform(audio):
    time = librosa.times_like(audio)

    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    st.pyplot()


def plot_mfcc(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')
    plt.tight_layout()
    st.pyplot()


def record_audio(duration):
    sample_rate = 44100
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio, sample_rate


def app():
    st.markdown("<h1 style='text-align: center;'>Parkinson Analysis</h1>", unsafe_allow_html=True)

    st.write("Record your voice and get a voice report using the Parselmouth library.")

    duration = st.slider("Recording duration (seconds):", min_value=1, max_value=10, value=4, step=1)

    if st.button("Record"):
        st.write("Recording...")
        audio, sample_rate = record_audio(duration)
        sf.write("input.wav", audio, sample_rate)
        st.write("Recorded!")
        st.write("Click on 'Play' to listen to the recorded audio.")

    if st.button("Play"):
        audio, sr = librosa.load("input.wav")
        Audio(audio, rate=sr)

    if st.button("Show Waveform"):
        audio, sr = librosa.load("input.wav")
        plot_waveform(audio)

    if st.button("Show MFCC"):
        audio, sr = librosa.load("input.wav")
        plot_mfcc(audio)

    if st.button("Show Voice Report"):
        voice_report = generate_voice_report("input.wav")
        st.write("Voice Report:")
        for line in voice_report:
            st.write(line)


if __name__ == '__main__':
    app()
