# logistic_regression_fixed.py

import pandas as pd
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
# FEATURE SCALING (IMPORTANT FIX)
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
model = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

# ==========================================
# TRAIN MODEL
# ==========================================
print("\nTraining Logistic Regression Model...")
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
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and Scaler Saved Successfully")

# ==========================================
# GRAPH (Accuracy)
# ==========================================
# ==========================================
# GRAPH (Accuracy)
# ==========================================

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Logistic Regression"],
    [accuracy * 100],
    color="red",      # Blue color
    edgecolor="black",
    linewidth=0.2,
    width=0.1
)

# Display accuracy value on top of bar
plt.text(
    0,
    accuracy * 100 + 1,
    f"{accuracy * 100:.2f}%",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.title(
    "Logistic Regression Accuracy",
    fontsize=14,
    fontweight="bold"
)

plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.ylim(0, 100)

plt.tight_layout()

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