import joblib
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset to train models matching the feature space of test_data.csv
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Train-Test Split (same random state to ensure consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Preprocessing (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create model directory and save scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# 3. Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("Training models...")
for name, model in models.items():
    if name in ["Logistic Regression", "kNN"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
        
    # Save model file
    filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    print(f"Saved: {filename}")

print("\nAll models trained and saved successfully!")