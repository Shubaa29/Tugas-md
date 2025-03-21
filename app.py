import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ObesityPredictionApp:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()

    def preprocess_data(self):
        # Encoding categorical data
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.encoder.fit_transform(self.data[col])

        # Splitting data into features and target
        X = self.data.drop('NObeyesdad', axis=1)
        y = self.data['NObeyesdad']

        # Normalizing numerical data
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

    def predict(self, input_data):
        input_data = np.array(input_data).reshape(1, -1)
        prediction = self.model.predict(input_data)
        probabilities = self.model.predict_proba(input_data)
        return prediction, probabilities

    def run(self):
        st.title("Obesity Level Prediction")
        
        # Display raw data
        if st.checkbox("Show Raw Data"):
            st.write(self.data)

        # Data Visualization
        if st.checkbox("Show Data Visualization"):
            st.write("### Correlation Heatmap")
            try:
                corr_matrix = self.data.corr()
                st.write("Correlation Matrix:", corr_matrix)  # Debugging: Display the correlation matrix
                if corr_matrix.isnull().any().any():
                    st.error("Correlation matrix contains NaN values. Check your data.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")

        # Input data numerik
        st.sidebar.header("Input Numerical Data")
        numerical_inputs = {}
        for col in self.X_train.select_dtypes(include=['float64', 'int64']).columns:
            numerical_inputs[col] = st.sidebar.slider(col, float(self.data[col].min()), float(self.data[col].max()), float(self.data[col].mean()))

        # Input data kategorikal
        st.sidebar.header("Input Categorical Data")
        categorical_inputs = {}
        for col in self.X_train.select_dtypes(include=['object']).columns:
            categorical_inputs[col] = st.sidebar.selectbox(col, self.data[col].unique())

        # Combine inputs
        input_data = []
        for col in self.X_train.columns:
            if col in numerical_inputs:
                input_data.append(numerical_inputs[col])
            else:
                input_data.append(categorical_inputs[col])

        # Display user input
        if st.checkbox("Show User Input"):
            st.write("### User Input Data")
            st.write(pd.DataFrame([input_data], columns=self.X_train.columns))

        # Predict and display results
        if st.button("Predict"):
            prediction, probabilities = self.predict(input_data)
            st.write("### Prediction Result")
            st.write(f"Predicted Obesity Level: {prediction[0]}")
            st.write("### Prediction Probabilities")
            st.write(pd.DataFrame(probabilities, columns=self.model.classes_))

# Load data
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Run the app
app = ObesityPredictionApp(data)
app.train_model()
app.run()
