import joblib
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Load data
df = pd.read_csv("SuperKart.csv")

X = df.drop("Product_Store_Sales_Total", axis=1)
y = df["Product_Store_Sales_Total"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

num = X.select_dtypes(include=["float64", "int64"]).columns
cat = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor())
])

model.fit(X_train, y_train)

joblib.dump(model, "final_model.pkl")
print("Model trained and saved!")
