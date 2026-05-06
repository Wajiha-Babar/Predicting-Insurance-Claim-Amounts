# src/train_model.py

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# 1. Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "medical_insurance.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# -------------------------------
# 2. Load Dataset
# -------------------------------

df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())


# -------------------------------
# 3. Target Column
# -------------------------------

TARGET = "annual_medical_cost"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")

# For assignment wording, we can also create charges alias
df["charges"] = df[TARGET]


# -------------------------------
# 4. Select Features
# -------------------------------

numeric_features = [
    "age",
    "income",
    "bmi",
    "visits_last_year",
    "hospitalizations_last_3yrs",
    "medication_count",
    "deductible",
    "copay",
    "risk_score",
    "chronic_count",
    "hypertension",
    "diabetes",
    "had_major_procedure"
]

categorical_features = [
    "sex",
    "region",
    "urban_rural",
    "education",
    "smoker",
    "plan_type",
    "network_tier"
]

features = numeric_features + categorical_features

# Keep only available columns
features = [col for col in features if col in df.columns]
numeric_features = [col for col in numeric_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]

df_model = df[features + [TARGET]].copy()


# -------------------------------
# 5. Data Cleaning
# -------------------------------

# Fill numeric missing values with median
for col in numeric_features:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# Fill categorical missing values with mode
for col in categorical_features:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

# Remove target missing values
df_model = df_model.dropna(subset=[TARGET])


# -------------------------------
# 6. Train-Test Split
# -------------------------------

X = df_model[features]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# 7. Preprocessing + Model Pipeline
# -------------------------------

def create_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", create_one_hot_encoder(), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# -------------------------------
# 8. Train Model
# -------------------------------

model.fit(X_train, y_train)


# -------------------------------
# 9. Predictions
# -------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# -------------------------------
# 10. Save Model and Metrics
# -------------------------------

joblib.dump(model, MODEL_DIR / "insurance_model.pkl")

metrics = {
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2_Score": round(r2, 4),
    "Training_Rows": int(X_train.shape[0]),
    "Testing_Rows": int(X_test.shape[0]),
    "Target": TARGET,
    "Model": "Linear Regression"
}

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

feature_options = {}

for col in categorical_features:
    feature_options[col] = sorted(df_model[col].dropna().unique().tolist())

for col in numeric_features:
    feature_options[col] = {
        "min": float(df_model[col].min()),
        "max": float(df_model[col].max()),
        "median": float(df_model[col].median())
    }

with open(MODEL_DIR / "feature_options.json", "w") as f:
    json.dump(feature_options, f, indent=4)


print("\nModel trained successfully!")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
print("Model saved in models/insurance_model.pkl")