import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
