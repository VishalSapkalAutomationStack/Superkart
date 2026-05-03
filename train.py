import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("SuperKart.csv")

TARGET = "Product_Store_Sales_Total"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------------------
# Feature types
# -------------------------------
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# -------------------------------
# Preprocessing
# -------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -------------------------------
# Model (BEST from your project)
# -------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=308,
        learning_rate=0.14,
        max_depth=8,
        random_state=123
    ))
])

# -------------------------------
# Train model
# -------------------------------
model.fit(X, y)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, "final_model.pkl")

print("Model trained and saved successfully!")
