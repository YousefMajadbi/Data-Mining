#!/usr/bin/env python3import pickle
import pandas as pd
import pickle
from train import extract_features

test_data = pd.read_csv("test.csv", header=None).values

X_test = extract_features(test_data)

with open('meal_no_meal_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test = scaler.fit_transform(X_test)

y_pred = model.predict(X_test)

# Save the predictions to Result.csv
pd.DataFrame(y_pred.astype(int)).to_csv('Result.csv', header=False, index=False)
