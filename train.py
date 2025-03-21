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
