"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import predict


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">K-Nearest Neighbour</b> for the Prediction of Parkinson's disease.
            </p>
        """, unsafe_allow_html=True)
    with st.expander("View attribute details"):
        st.markdown("""Sure, here is a brief description of each feature:

- Median pitch: The median value of the fundamental frequency (pitch) of the voice samples, which indicates the 
central tendency of the pitch distribution.

- Mean pitch: The mean value of the fundamental frequency (pitch) of the voice samples, which indicates the average 
pitch of the voice.

- Standard deviation: A measure of the variability of the fundamental frequency (pitch) of the voice samples, 
which indicates how spread out the pitch values are from the mean.

- Minimum pitch: The lowest fundamental frequency (pitch) value of the voice samples.

- Maximum pitch: The highest fundamental frequency (pitch) value of the voice samples.

- Number of pulses: The number of distinct pulses in the voice sample waveform, which can be used to estimate the 
speaking rate.

- Number of periods: The number of distinct periods in the voice sample waveform, which can be used to estimate the 
pitch.

- Mean period: The mean value of the period of the voice samples, which is the time between consecutive pulses.

- Standard deviation of period: A measure of the variability of the period of the voice samples.

- Fraction of locally unvoiced frames: The proportion of frames in the voice sample that are classified as unvoiced (
i.e., no periodicity).

- Number of voice breaks: The number of abrupt transitions between voiced and unvoiced frames in the voice sample.

- Degree of voice breaks: The percentage of time in the voice sample that is characterized by voice breaks.

- Jitter (local): A measure of the variation in the period of consecutive voice periods, calculated as a percentage 
of the period of the current voice period.

- Jitter (local, absolute): A measure of the variation in the period of consecutive voice periods, expressed in seconds.

- Jitter (rap): A measure of the variation in the period of consecutive voice periods, calculated as a percentage of 
the period of the previous voice period.

- Jitter (ppq5): A measure of the variation in the period of consecutive voice periods, calculated as a percentage of 
the period of the previous five voice periods.

- Jitter (ddp): A measure of the variation in the period of consecutive voice periods, calculated as the difference 
between the absolute values of the differences between successive periods.

- Shimmer (local): A measure of the variation in the amplitude of consecutive voice periods, expressed as a 
percentage of the mean amplitude of the current voice period.

- Shimmer (local, dB): A measure of the variation in the amplitude of consecutive voice periods, expressed in decibels.

- Shimmer (apq3): A measure of the variation in the amplitude of consecutive voice periods, calculated as a 
percentage of the mean amplitude of the previous three voice periods.

- Shimmer (apq5): A measure of the variation in the amplitude of consecutive voice periods, calculated as a 
percentage of the mean amplitude of the previous five voice periods.

- Shimmer (apq11): A measure of the variation in the amplitude of consecutive voice periods, calculated as a 
percentage of the mean amplitude of the previous 11 voice periods.

- Shimmer (dda): A measure of the variation in the amplitude of consecutive voice periods, calculated as the absolute 
difference between the maxima and minima of the waveform.

- Mean autocorrelation: A measure of the periodicity of the voiced parts of the voice sample, calculated as the 
average correlation coefficient between the waveform and a delayed version of itself.

- Mean noise-to-harmonics ratio: A measure of the noise in the voiced parts of the voice sample relative to the 
harmonics, expressed as a ratio.

- Mean harmonics-to-noise ratio: A measure of the harmonics in the voiced parts of the voice sample relative to the 
noise,""")
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    Jitter_local = st.slider("Jitter_local", int(df["Jitter_local"].min()), int(df["Jitter_local"].max()))
    Jitter_local_absolute = st.slider("Jitter_local_absolute", int(df["Jitter_local_absolute"].min()),
                                      int(df["Jitter_local_absolute"].max()))
    Jitter_rap = st.slider("Jitter_rap", int(df["Jitter_rap"].min()), int(df["Jitter_rap"].max()))
    Jitter_ppq5 = st.slider("Jitter_ppq5", float(df["Jitter_ppq5"].min()), float(df["Jitter_ppq5"].max()))
    Jitter_ddp = st.slider("Jitter_ddp", float(df["Jitter_ddp"].min()), float(df["Jitter_ddp"].max()))
    Shimmer_local_dB = st.slider("Shimmer_local_dB", float(df["Shimmer_local_dB"].min()),
                                 float(df["Shimmer_local_dB"].max()))
    Shimmer_apq3 = st.slider("Shimmer_apq3", float(df["Shimmer_apq3"].min()), float(df["Shimmer_apq3"].max()))
    Shimmer_apq5 = st.slider("Shimmer_apq5", float(df["Shimmer_apq5"].min()), float(df["Shimmer_apq5"].max()))
    Shimmer_apq11 = st.slider("Shimmer_apq11", float(df["Shimmer_apq11"].min()), float(df["Shimmer_apq11"].max()))
    Shimmer_dda = st.slider("Shimmer_dda", float(df["Shimmer_dda"].min()), float(df["Shimmer_dda"].max()))
    Median_pitch = st.slider("Median_pitch", float(df["Median_pitch"].min()), float(df["Median_pitch"].max()))
    Mean_pitch = st.slider("Mean_pitch", float(df["Mean_pitch"].min()), float(df["Mean_pitch"].max()))
    Minimum_pitch = st.slider("Minimum_pitch", float(df["Minimum_pitch"].min()), float(df["Minimum_pitch"].max()))
    Maximum_pitch = st.slider("Maximum_pitch", float(df["Maximum_pitch"].min()), float(df["Maximum_pitch"].max()))
    Number_of_pulses = st.slider("Number_of_pulses", float(df["Number_of_pulses"].min()),
                                 float(df["Number_of_pulses"].max()))
    Number_of_periods = st.slider("Number_of_periods", float(df["Number_of_periods"].min()),
                                  float(df["Number_of_periods"].max()))
    Mean_period = st.slider("Mean_period", float(df["Mean_period"].min()), float(df["Mean_period"].max()))
    Standard_deviation = st.slider("Standard_deviation", float(df["Standard_deviation"].min()),
                                   float(df["Standard_deviation"].max()))
    Standard_deviation_of_period = st.slider("Standard_deviation_of_period",
                                             float(df["Standard_deviation_of_period"].min()),
                                             float(df["Standard_deviation_of_period"].max()))
    Number_of_voice_breaks = st.slider("Number_of_voice_breaks", float(df["Number_of_voice_breaks"].min()),
                                       float(df["Number_of_voice_breaks"].max()))
    Fraction_of_locally_unvoiced_frames = st.slider("Fraction_of_locally_unvoiced_frames",
                                                    float(df["Fraction_of_locally_unvoiced_frames"].min()),
                                                    float(df["Fraction_of_locally_unvoiced_frames"].max()))
    Degree_of_voice_breaks = st.slider("Degree_of_voice_breaks", float(df["Degree_of_voice_breaks"].min()),
                                       float(df["Degree_of_voice_breaks"].max()))
    UPDRS = st.slider("UPDRS", float(df["UPDRS"].min()), float(df["UPDRS"].max()))

    # Create a list to store all the features
    features = [Jitter_local, Jitter_local_absolute, Jitter_rap, Jitter_ppq5, Jitter_ddp, Shimmer_local_dB, Shimmer_apq3, Shimmer_apq5, Shimmer_apq11, Shimmer_dda, Median_pitch,
                Mean_pitch, Standard_deviation, Minimum_pitch, Maximum_pitch, Number_of_pulses,
                Number_of_periods, Mean_period, Standard_deviation_of_period, Fraction_of_locally_unvoiced_frames,
                Number_of_voice_breaks, Degree_of_voice_breaks, UPDRS]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        st.success("Predicted Sucessfully")

        # Print the output according to the prediction
        if prediction == 1:
            st.warning("The person either has Parkison's disease or prone to get Parkinson's disease")
        else:
            st.info("The person is safe from Parkinson's disease")

        # Print teh score of the model 
        st.write("The model used is trusted by doctor and has an accuracy of ", (score * 100), "%")
