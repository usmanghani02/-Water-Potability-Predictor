# model_trainer.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("df/water_potability.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Preprocessing + XGBoost
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=3))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "improved_xgboost_model.joblib")
print("✅ Model saved as improved_xgboost_model.joblib")
