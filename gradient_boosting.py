# gradient_model.py

import pandas as pd
import warnings
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("diabetes_prediction_project\cleaned_diabetes.csv")

print("Dataset Loaded")

# ==========================================
# DATA CLEANING
# ==========================================

cols = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

for col in cols:
    data[col] = data[col].replace(0, data[col].median())

print("Data Cleaned")

# ==========================================
# REMOVE OUTLIERS
# ==========================================

for col in cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower) & (data[col] <= upper)]

print("Outliers Removed")

# ==========================================
# SPLIT
# ==========================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling Done")

# ==========================================
# SMOTE
# ==========================================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("SMOTE Applied")

# ==========================================
# MODEL
# ==========================================

print("\nTraining Gradient Boosting Model...")

model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# EVALUATION
# ==========================================

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"\nAccuracy: {round(accuracy * 100, 2)} %")

# ==========================================
# SAVE MODEL
# ==========================================

if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(model, open("models/gradient_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Model Saved")

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_diabetes(input_data):
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    model = pickle.load(open("models/gradient_model.pkl", "rb"))

    input_scaled = scaler.transform([input_data])
    result = model.predict(input_scaled)[0]

    if result == 1:
        return "Diabetic"
    else:
        return "Not Diabetic"

# ==========================================
# SAMPLE TEST
# ==========================================

sample = [6,148,72,35,125,33.6,0.627,50]

result = predict_diabetes(sample)

print("\nSample Prediction:", result)