# random_forest_fixed.py

import pandas as pd
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# REMOVE WARNINGS
# ==========================================
warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATASET
# ==========================================
data = pd.read_csv("diabetes_prediction_project\cleaned_diabetes.csv")

print("====================================")
print("Dataset Loaded Successfully")
print("====================================")
print(data.head())

# ==========================================
# INPUT / OUTPUT
# ==========================================
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ==========================================
# FEATURE SCALING (GOOD PRACTICE)
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# SPLIT DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================================
# CREATE MODEL
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================
print("\nTraining Random Forest Model...")
model.fit(X_train, y_train)
print("Model Trained Successfully")

# ==========================================
# TEST MODEL
# ==========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("Model Accuracy =", round(accuracy * 100, 2), "%")
print("====================================")

# ==========================================
# SAVE MODEL + SCALER
# ==========================================
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and Scaler Saved Successfully")

# ==========================================
# GRAPH (Accuracy Visualization)
# ==========================================
plt.figure()
plt.bar(["Random Forest"], [accuracy * 100])
plt.ylabel("Accuracy (%)")
plt.title("Random Forest Accuracy")
plt.show()

# ==========================================
# SINGLE PREDICTION (FIXED)
# ==========================================
sample_data = np.array([[
    6, 148, 72, 35, 0, 33.6, 0.627, 50
]])

# SCALE INPUT
sample_scaled = scaler.transform(sample_data)

prediction = model.predict(sample_scaled)

print("\n====================================")
print("Prediction Result =", prediction[0])
print("====================================")

if prediction[0] == 1:
    print("Diabetes Positive")
else:
    print("Diabetes Negative")