"""
train_models.py
----------------
Trains 5 classification models (Logistic Regression, Decision Tree, kNN,
Naive Bayes, Random Forest) on the Breast Cancer Wisconsin dataset,
evaluates each with Accuracy, AUC, Precision, Recall, F1, and MCC,
saves the trained models + scaler for the Streamlit app, and saves a
held-out test set (test_data.csv) for grading/demo purposes.

Run:  python model/train_models.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data           # 30 numeric features
y = data.target         # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}, classes: {dict(pd.Series(y).value_counts())}")

# ---------------------------------------------------------------------
# 2. Train / test split (stratified so class balance is preserved)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the RAW (unscaled) test data + true labels -> this is the CSV
# students upload into the Streamlit app and also submit to GitHub.
test_export = X_test.copy()
test_export["target"] = y_test.values
test_export.to_csv("test_data.csv", index=False)
print("Saved test_data.csv:", test_export.shape)

# ---------------------------------------------------------------------
# 3. Scale features (kNN, Logistic Regression benefit from this)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.pkl")

# ---------------------------------------------------------------------
# 4. Define the 6 required models (Logistic Regression, Decision Tree,
#    kNN, Naive Bayes, Random Forest = 5 distinct algorithms; the
#    assignment table lists them as 5 rows even though it says "6" in
#    the intro paragraph -- follow the table in Step 5, which has 5 rows)
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
}

results = []
for name, model in models.items():
    # kNN and Logistic Regression train on scaled data; tree-based
    # models don't need scaling but scaled data doesn't hurt them.
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    # Save each trained model for the Streamlit app to load later
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"model/{safe_name}.pkl")
    print(f"Trained {name}: {metrics}")

# ---------------------------------------------------------------------
# 5. Save comparison table (used directly in README.md and by app.py)
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model/model_comparison.csv", index=False)

with open("model/feature_names.json", "w") as f:
    json.dump(list(X.columns), f)

print("\n=== FINAL COMPARISON TABLE ===")
print(results_df.to_string(index=False))
print("\nAll models, scaler, and test_data.csv saved successfully.")
