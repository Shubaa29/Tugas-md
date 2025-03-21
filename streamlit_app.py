import streamlit as st


class Preprocessing:
    def __init__(self, data):
        self.data = data

    def encode_categorical_features(self):
        # Encode categorical features
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            self.data[col] = label_encoder.fit_transform(self.data[col])
        return self.data

    def normalize_numerical_features(self):
        # Normalize numerical features
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
        return self.data

    def preprocess_data(self):
        self.encode_categorical_features()
        self.normalize_numerical_features()
        return self.data


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ObesityModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()

    def split_data(self):
        X = self.data.drop('NObeyesdad', axis=1)
        y = self.data['NObeyesdad']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

import pandas as pd
from obesity_prediction.preprocessing import Preprocessing
from obesity_prediction.model import ObesityModel

def main():
    # Load data
    data = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')

    # Preprocess data
    preprocessor = Preprocessing(data)
    processed_data = preprocessor.preprocess_data()

    # Train model
    obesity_model = ObesityModel(processed_data)
    obesity_model.split_data()
    obesity_model.train_model()
    accuracy = obesity_model.evaluate_model()

    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
from obesity_prediction.preprocessing import Preprocessing
from obesity_prediction.model import ObesityModel

def main():
    st.title("Obesity Prediction App")

    # Load data
    data = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')

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
