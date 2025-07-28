from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Classification Model
def train_classifier():
    data = pd.read_csv('features_pump.csv')
    X = data.drop(['timestamp', 'machine_status', 'failure_flag'], axis=1)
    y = data['failure_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'failure_classifier.pkl')

# Regression Model (RUL)
def train_regressor():
    data = pd.read_csv('features_nasa.csv')
    X = data.drop(['id', 'RUL', 'cycle'], axis=1)
    y = data['RUL']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, 'rul_predictor.pkl')

if __name__ == "__main__":
    train_classifier()
    train_regressor()