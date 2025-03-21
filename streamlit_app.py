import streamlit as st
import pandas as pd
import numpy as np


class ObesityModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
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
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        return acc, classification_report(y_test, predictions)

    def predict(self, input_data):
        input_data_scaled = self.scaler.transform([input_data])
        prediction = self.model.predict(input_data_scaled)[0]
        probabilities = self.model.predict_proba(input_data_scaled)[0]
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
input_data = []

# Numerical Inputs
for col in model.data.select_dtypes(include=['float64', 'int64']).columns:
    value = st.slider(f"{col}", float(model.data[col].min()), float(model.data[col].max()), float(model.data[col].mean()))
    input_data.append(value)

# Categorical Inputs
for col in model.data.select_dtypes(include=['object']).columns:
    options = list(model.label_encoders[col].classes_)
    value = st.selectbox(f"{col}", options)
    input_data.append(model.label_encoders[col].transform([value])[0])

# Show input data
st.write("### Input Data Preview")
st.write(input_data)

# Prediction
if st.button("Predict"):
    prediction, probabilities = model.predict(input_data)
    st.write(f"### Predicted Class: {prediction}")
    st.write(f"### Probability per Class: {probabilities}")
