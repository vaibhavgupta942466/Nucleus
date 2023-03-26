import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import joblib

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Split the dataset into features (X) and target (y)
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT'].copy()

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1, random_state=13)

class RFCmodel:
    def __init__(self):
        # Train a Random Forest Classifier model
        self.model = RandomForestClassifier(min_samples_split=2, class_weight={0:2,1:7}, random_state=13)
        self.model.fit(train_X,train_y)
        self.prediction=self.model.predict(test_X)
        self.accuracy = accuracy_score(self.prediction, test_y)

    def predict(self, user_input):
        return self.model.predict(user_input)

class DTCmodel:
    def __init__(self):
        # Train a Decision Tree Classifier model
        self.model = DecisionTreeClassifier(min_samples_split=2, class_weight={0:2,1:7}, random_state=13)
        self.model.fit(train_X,train_y)
        self.prediction=self.model.predict(test_X)
        self.accuracy = accuracy_score(self.prediction, test_y)

    def predict(self, user_input):
        return self.model.predict(user_input)
        
# Save the trained models
joblib.dump(DTCmodel(), 'DTCmodel.joblib')
joblib.dump(RFCmodel(), 'RFCmodel.joblib')
