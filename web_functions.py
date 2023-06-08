"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


@st.cache_resource()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('pd1.csv')

    df = df.rename(columns={
        "Jitter (local)": "Jitter_local",
        "Jitter (local, absolute)": "Jitter_local_absolute",
        "Jitter (rap)": "Jitter_rap",
        "Jitter (ppq5)": "Jitter_ppq5",
        "Jitter (ddp)": "Jitter_ddp",
        "Shimmer (local, dB)": "Shimmer_local_dB",
        "Shimmer (apq3)": "Shimmer_apq3",
        "Shimmer (apq5)": "Shimmer_apq5",
        "Shimmer (apq11)": "Shimmer_apq11",
        "Shimmer (dda)": "Shimmer_dda",
        "Median pitch": "Median_pitch",
        "Mean pitch": "Mean_pitch",
        "Standard deviation": "Standard_deviation",
        "Minimum pitch": "Minimum_pitch",
        "Maximum pitch": "Maximum_pitch",
        "Number of pulses": "Number_of_pulses",
        "Number of periods": "Number_of_periods",
        "Mean period": "Mean_period",
        "Standard deviation of period": "Standard_deviation_of_period",
        "Fraction of locally unvoiced frames": "Fraction_of_locally_unvoiced_frames",
        "Number of voice breaks": "Number_of_voice_breaks",
        "Degree of voice breaks": "Degree_of_voice_breaks",
        "UPDRS": "UPDRS"
    })

    # Perform feature and target split
    X = df[["Jitter_local", "Jitter_local_absolute", "Jitter_rap", "Jitter_ppq5", "Jitter_ddp",
            "Shimmer_local_dB", "Shimmer_apq3", "Shimmer_apq5", "Shimmer_apq11", "Shimmer_dda", "Median_pitch",
            "Mean_pitch", "Standard_deviation", "Minimum_pitch", "Maximum_pitch", "Number_of_pulses",
            "Number_of_periods", "Mean_period", "Standard_deviation_of_period", "Fraction_of_locally_unvoiced_frames",
            "Number_of_voice_breaks", "Degree_of_voice_breaks", "UPDRS"]]
    y = df['status']

    return df, X, y


@st.cache_resource()
def train_model(X, y):
    """This function trains the model and returns the model and model score"""
    # Create the model
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], activation='relu'))  # Example hidden layer with 16 neurons and ReLU activation
    model.add(Dense(1, activation='sigmoid'))  # Example output layer with 1 neuron and sigmoid activation

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)

    # Evaluate the model
    _, accuracy = model.evaluate(X, y)

    # Return the trained model and model score
    return model, accuracy

  
def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
