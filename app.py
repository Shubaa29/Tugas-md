import streamlit as st
import pandas as pd
import sys
import os

# Tambahkan path ke folder root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from obesity_prediction.preprocessing import Preprocessing
from obesity_prediction.model import ObesityModel

def main():
    st.title("Obesity Prediction App")

    # Load data
    data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    # Preprocess data
    preprocessor = Preprocessing(data)
    processed_data = preprocessor.preprocess_data()

    # Train model
    obesity_model = ObesityModel(processed_data)
    obesity_model.split_data()
    obesity_model.train_model()

    # Display accuracy
    accuracy = obesity_model.evaluate_model()
    st.write(f"Model Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
