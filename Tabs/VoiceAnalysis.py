import streamlit as st
import parselmouth
import numpy as np
from parselmouth.praat import call
import pickle
import tempfile


loaded_model = pickle.load(open('rr.pickle', 'rb'))


# Function to analyze audio and generate voice report
def analyze_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(audio_file.read())
        temp_path = temp.name

    sound = parselmouth.Sound(temp_path)
    pitch = call(sound, "To Pitch", 0.0, 20, 100)
    pulse = call([sound, pitch], "To PointProcess (cc)")
    voice_report = parselmouth.praat.call([sound, pitch, pulse], "Voice report", 0.0, sound.get_total_duration(), 75,
                                          600, 1.3, 1.6, 0.03, 0.45).split("\n")

    return voice_report


# Function to save voice report to file
def save_report_to_file(voice_report):
    with open("voice_report.txt", "w") as f:
        for line in voice_report:
            f.write(line + "\n")


# Function to extract data from the voice report
def extract_data_from_report(report_content):
    data = {}

    # Pitch
    pitch_start = report_content.find('Pitch:') + len('Pitch:\n')
    pitch_end = report_content.find('Pulses:')
    pitch_data = report_content[pitch_start:pitch_end].split('\n')
    data['Median pitch'] = float(pitch_data[0].split(':')[1].strip().split()[0])
    data['Mean pitch'] = float(pitch_data[1].split(':')[1].strip().split()[0])
    data['Standard deviation'] = float(pitch_data[2].split(':')[1].strip().split()[0])
    data['Minimum pitch'] = float(pitch_data[3].split(':')[1].strip().split()[0])
    data['Maximum pitch'] = float(pitch_data[4].split(':')[1].strip().split()[0])

    # Pulses
    pulses_start = report_content.find('Pulses:') + len('Pulses:\n')
    pulses_end = report_content.find('Voicing:')
    pulses_data = report_content[pulses_start:pulses_end].split('\n')
    data['Number of pulses'] = int(pulses_data[0].split(':')[1].strip())
    data['Number of periods'] = int(pulses_data[1].split(':')[1].strip())
    data['Mean period'] = float(pulses_data[2].split(':')[1].strip().split()[0])
    data['Standard deviation of period'] = float(pulses_data[3].split(':')[1].strip().split()[0])

    # Voicing
    voicing_start = report_content.find('Voicing:') + len('Voicing:\n')
    voicing_end = report_content.find('Jitter:')
    voicing_data = report_content[voicing_start:voicing_end].split('\n')
    data['Fraction of locally unvoiced frames'] = float(voicing_data[0].split(':')[1].strip().split('%')[0]) / 100
    data['Number of voice breaks'] = int(voicing_data[1].split(':')[1].strip())
    data['Degree of voice breaks'] = float(voicing_data[2].split(':')[1].strip().split('%')[0]) / 100

    # Jitter
    jitter_start = report_content.find('Jitter:') + len('Jitter:\n')
    jitter_end = report_content.find('Shimmer:')
    jitter_data = report_content[jitter_start:jitter_end].split('\n')
    data['Jitter (local)'] = float(jitter_data[0].split(':')[1].strip().split('%')[0]) / 100
    data['Jitter (local, absolute)'] = float(jitter_data[1].split(':')[1].strip().split()[0])
    data['Jitter (rap)'] = float(jitter_data[2].split(':')[1].strip().split('%')[0]) / 100
    data['Jitter (ppq5)'] = float(jitter_data[3].split(':')[1].strip().split('%')[0]) / 100
    data['Jitter (ddp)'] = float(jitter_data[4].split(':')[1].strip().split('%')[0]) / 100

    # Shimmer
    shimmer_start = report_content.find('Shimmer:') + len('Shimmer:\n')
    shimmer_end = report_content.find('Harmonicity of the voiced parts only:')
    shimmer_data = report_content[shimmer_start:shimmer_end].split('\n')
    data['Shimmer (local)'] = float(shimmer_data[0].split(':')[1].strip().split('%')[0]) / 100
    data['Shimmer (local, dB)'] = float(shimmer_data[1].split(':')[1].strip().split()[0])
    data['Shimmer (apq3)'] = float(shimmer_data[2].split(':')[1].strip().split('%')[0]) / 100
    data['Shimmer (apq5)'] = float(shimmer_data[3].split(':')[1].strip().split('%')[0]) / 100
    data['Shimmer (apq11)'] = shimmer_data[4].split(':')[1].strip()

    # Harmonicity of the voiced parts only
    harmonicity_start = report_content.find('Harmonicity of the voiced parts only:') + len(
        'Harmonicity of the voiced parts only:\n')
    harmonicity_data = report_content[harmonicity_start:].split('\n')
    data['Mean autocorrelation'] = float(harmonicity_data[0].split(':')[1].strip())
    data['Mean noise-to-harmonics ratio'] = float(harmonicity_data[1].split(':')[1].strip())
    data['Mean harmonics-to-noise ratio'] = float(harmonicity_data[2].split(':')[1].strip().split()[0])

    return data


# Main Streamlit app
def app():
    st.title("Parkinson's Disease Detection")

    # File uploader
    st.subheader("Upload an audio file:")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    if uploaded_file is not None:
        # Display the uploaded audio file
        st.subheader("Uploaded Audio File:")
        st.audio(uploaded_file)

        # Analyze the audio and generate the voice report
        voice_report = analyze_audio(uploaded_file)

        # Save the voice report to a file
        save_report_to_file(voice_report)

        # Read the saved report file
        with open('voice_report.txt', 'r') as file:
            report_content = file.read()

        # Extract data from the voice report
        data = extract_data_from_report(report_content)

        # Display the extracted data
        st.subheader("Voice Report:")
        for key, value in data.items():
            st.write(key + ':', value)

        # Perform prediction using the extracted data
        median_pitch = data['Median pitch']
        mean_pitch = data['Mean pitch']
        standard_deviation = data['Standard deviation']
        fraction_unvoiced = data['Fraction of locally unvoiced frames']
        number_of_pulses = data['Number of pulses']
        number_of_periods = data['Number of periods']
        mean_period = data['Mean period']
        std_dev_period = data['Standard deviation of period']
        number_of_voice_breaks = data['Number of voice breaks']
        degree_of_voice_breaks = data['Degree of voice breaks']
        jitter_local = data['Jitter (local)']
        jitter_local_absolute = data['Jitter (local, absolute)']

        # Combine the values into an input data array
        input_data = np.array([
            median_pitch, mean_pitch, standard_deviation,
            fraction_unvoiced, number_of_pulses, number_of_periods,
            mean_period, std_dev_period, number_of_voice_breaks,
            degree_of_voice_breaks, jitter_local, jitter_local_absolute,
        ])

        # Reshape the input data
        input_data_reshaped = input_data.reshape(1, -1)

        # Standardize the data (assuming you have a 'scaler' object trained on data with 22 features)
        prediction = None  # Initialize the variable with a default value

        if st.button("Predict"):
            prediction = loaded_model.predict(input_data_reshaped)

        # Display the prediction result
        st.subheader("Prediction:")
        if prediction is None:
            st.warning("No prediction made yet")
        elif prediction == 0:
            st.warning("The person either has Parkinson's disease or is prone to getting Parkinson's disease")
        else:
            st.info("The person is safe from Parkinson's disease")


if __name__ == '__main__':
    app()