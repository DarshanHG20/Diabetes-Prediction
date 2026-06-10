import pandas as pd
import numpy as np
import pickle
import warnings
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv(r"diabetes_prediction_project\cleaned_diabetes.csv")

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

# ==========================================
# DATA CLEANING
# ==========================================

cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in cols:
    data[col] = data[col].replace(0, data[col].median())

print("Data Cleaning Completed")

# ==========================================
# REMOVE OUTLIERS
# ==========================================

for col in cols:

    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    data = data[
        (data[col] >= lower) &
        (data[col] <= upper)
    ]

print("Outliers Removed")

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling Completed")

# ==========================================
# APPLY SMOTE
# ==========================================

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE Applied")

# ==========================================
# CREATE MODEL FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# MODELS
# ==========================================

models = {

    "Random Forest":
    RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42
    ),

    "Extra Trees":
    ExtraTreesClassifier(
        n_estimators=300,
        max_depth=25,
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

best_model = None
best_accuracy = 0
best_name = ""

accuracies = []
names = []

# ==========================================
# TRAIN MODELS
# ==========================================

for name, model in models.items():

    print(f"\nTraining {name} ...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"{name} Accuracy : {acc * 100:.2f}%")

    accuracies.append(acc * 100)
    names.append(name)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# ==========================================
# SAVE BEST MODEL
# ==========================================

pickle.dump(
    best_model,
    open("models/best_model.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

print("\n" + "=" * 50)
print("BEST MODEL :", best_name)
print("BEST ACCURACY :", round(best_accuracy * 100, 2), "%")
print("=" * 50)

# ==========================================
# MODEL COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

colors = [
    "#E74C3C",   # Red
    "#2ECC71",   # Green
    "#3498DB"    # Blue
]

bars = plt.bar(
    names,
    accuracies,
    color=colors,
    edgecolor="black",
    linewidth=2
)

best_index = names.index(best_name)

bars[best_index].set_edgecolor("gold")
bars[best_index].set_linewidth(4)

plt.xlabel("Machine Learning Models", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title(
    "Diabetes Prediction Model Comparison",
    fontsize=16,
    fontweight="bold"
)

for i, v in enumerate(accuracies):
    plt.text(
        i,
        v + 0.5,
        f"{v:.2f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)

plt.show()

print("Graph Saved Successfully")

# ==========================================
# PATIENT PREDICTION
# ==========================================

print("\nENTER PATIENT DETAILS")

preg = float(input("Pregnancies : "))
glucose = float(input("Glucose : "))
bp = float(input("Blood Pressure : "))
skin = float(input("Skin Thickness : "))
insulin = float(input("Insulin : "))
bmi = float(input("BMI : "))
dpf = float(input("Diabetes Pedigree Function : "))
age = float(input("Age : "))

patient = np.array([
    preg,
    glucose,
    bp,
    skin,
    insulin,
    bmi,
    dpf,
    age
]).reshape(1, -1)

patient_scaled = scaler.transform(patient)

prediction = best_model.predict(patient_scaled)[0]

probability = best_model.predict_proba(patient_scaled)[0]

print("\n" + "=" * 50)
print("DIABETES PREDICTION RESULT")
print("=" * 50)

if prediction == 1:
    print("PATIENT STATUS : DIABETIC")
    print(
        "DIABETES PROBABILITY :",
        round(probability[1] * 100, 2),
        "%"
    )
else:
    print("PATIENT STATUS : NOT DIABETIC")
    print(
        "CONFIDENCE :",
        round(probability[0] * 100, 2),
        "%"
    )

print("=" * 50)