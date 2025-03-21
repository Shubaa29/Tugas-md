import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_model(filename):
  model = joblib.load(filename)
  return model

def predict_with_model(model, user_input):
  prediction = model.predict([user_input])
  return prediction[0]

class ObesityModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()
    
    def preprocess_data(self):
        # Encoding categorical variables
        for col in self.data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le
        
        # Splitting features and target
        X = self.data.drop(columns=['NObeyesdad'])  # Target column
        y = self.data['NObeyesdad']
        
        # Normalization
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        return acc, classification_report(self.y_test, predictions)

    def preprocess_input(self, input_data):
        input_df = pd.DataFrame([input_data], columns=self.data.drop(columns=['NObeyesdad']).columns)
        for col, le in self.label_encoders.items():
            input_df[col] = le.transform(input_df[col])
        input_scaled = self.scaler.transform(input_df)
        return input_scaled
    
    def predict(self, input_data):
        input_scaled = self.preprocess_input(input_data)
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        return prediction, probabilities

# Streamlit App
st.title("Obesity Classification with Random Forest")
model = ObesityModel('/mnt/data/ObesityDataSet_raw_and_data_sinthetic.csv')
acc, report = model.train()

st.write(f"### Model Accuracy: {acc}")
st.text(report)

st.write("### Raw Data")
st.dataframe(model.data.head())

st.write("### Input User Data")
input_data = {}

# Numerical Inputs
for col in model.data.select_dtypes(include=['float64', 'int64']).columns:
    input_data[col] = st.slider(f"{col}", float(model.data[col].min()), float(model.data[col].max()), float(model.data[col].mean()))

# Categorical Inputs
for col in model.label_encoders.keys():
    options = list(model.label_encoders[col].classes_)
    value = st.selectbox(f"{col}", options)
    input_data[col] = value

# Show input data
st.write("### Input Data Preview")
st.dataframe(pd.DataFrame([input_data]))

# Prediction
if st.button("Predict"):
    prediction, probabilities = model.predict(input_data)
    probability_df = pd.DataFrame([probabilities], columns=model.model.classes_)
    
    st.write(f"### Predicted Class: {prediction}")
    st.write("### Probability per Class:")
    st.dataframe(probability_df)
