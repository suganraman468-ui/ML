"""
Streamlit demo app for Assignment 2.

Features (per assignment requirement):
  a. Dataset upload option (CSV)          -> st.file_uploader
  b. Model selection dropdown             -> st.selectbox
  c. Display of evaluation metrics        -> metric cards + table
  d. Confusion matrix / classification report -> seaborn heatmap + text
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    with open("model/feature_names.json") as f:
        feature_names = json.load(f)
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, feature_names, models

scaler, feature_names, models = load_artifacts()

st.title("🔬 Breast Cancer Classification — Model Comparison Demo")
st.caption("BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning Assignment 2")

# -----------------------------------------------------------------
# a. Dataset upload
# -----------------------------------------------------------------
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (must include a 'target' column)", type=["csv"]
)

# -----------------------------------------------------------------
# b. Model selection dropdown
# -----------------------------------------------------------------
st.sidebar.header("2. Choose a Model")
model_name = st.sidebar.selectbox("Model", list(models.keys()))
model = models[model_name]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column with true labels (0/1).")
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols[:5]}...")
        st.stop()

    X = df[feature_names]
    y_true = df["target"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # -----------------------------------------------------------------
    # c. Display evaluation metrics
    # -----------------------------------------------------------------
    st.subheader(f"Evaluation Metrics — {model_name}")
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    # -----------------------------------------------------------------
    # d. Confusion matrix + classification report
    # -----------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Malignant (0)", "Benign (1)"],
                    yticklabels=["Malignant (0)", "Benign (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_right:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"])
        st.text(report)

    # Bonus: let user compare ALL models at once on the uploaded data
    st.subheader("Compare All Models on This Data")
    rows = []
    for name, m in models.items():
        pred = m.predict(X_scaled)
        proba = m.predict_proba(X_scaled)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, pred), 4),
            "AUC": round(roc_auc_score(y_true, proba), 4),
            "Precision": round(precision_score(y_true, pred), 4),
            "Recall": round(recall_score(y_true, pred), 4),
            "F1": round(f1_score(y_true, pred), 4),
            "MCC": round(matthews_corrcoef(y_true, pred), 4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

else:
    st.info("👈 Upload `test_data.csv` from the sidebar to see predictions and metrics.")
    st.markdown(
        "You can use the `test_data.csv` file generated by `model/train_models.py` "
        "(also included in this repository)."
    )
