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
