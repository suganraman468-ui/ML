import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="ML Assignment 2 - Classification App", layout="wide")

st.title("WILP M.Tech Machine Learning - Assignment 2")
st.write("**Interactive Classification Model Evaluation Dashboard**")

# Sidebar Configuration
st.sidebar.header("Configuration Panel")

# Model Selection Dropdown
model_options = ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
selected_model_name = st.sidebar.selectbox("Select Classification Model", model_options)

# Dataset Upload Option
st.sidebar.subheader("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your test_data.csv", type=["csv"])

# Load model and scaler helper
@st.cache_resource
def load_artifacts(model_name):
    file_map = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "kNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl"
    }
    model = joblib.load(file_map[model_name])
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts(selected_model_name)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run `train_models.py` first.")
    st.stop()

# Load test data (either uploaded or default test_data.csv)
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    try:
        test_df = pd.read_csv("test_data.csv")
    except FileNotFoundError:
        st.warning("Please upload `test_data.csv` using the sidebar or place it in the project root folder.")
        st.stop()

st.subheader(f"Evaluation Results for: `{selected_model_name}`")

if 'target' in test_df.columns:
    X_eval = test_df.drop(columns=['target'])
    y_eval = test_df['target']
    
    # Apply scaling for models trained on scaled data
    if selected_model_name in ["Logistic Regression", "kNN"]:
        X_eval_input = scaler.transform(X_eval)
    else:
        X_eval_input = X_eval
        
    # Predictions and probabilities
    predictions = model.predict(X_eval_input)
    try:
        probabilities = model.predict_proba(X_eval_input)[:, 1]
    except Exception:
        probabilities = np.zeros(len(predictions))
        
    # Calculate metrics
    acc = accuracy_score(y_eval, predictions)
    auc = roc_auc_score(y_eval, probabilities) if len(np.unique(y_eval)) > 1 else 0.0
    prec = precision_score(y_eval, predictions, zero_division=0)
    rec = recall_score(y_eval, predictions, zero_division=0)
    f1 = f1_score(y_eval, predictions, zero_division=0)
    mcc = matthews_corrcoef(y_eval, predictions)
    
    # Display metrics in 6 columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix & Classification Report Layout
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_eval, predictions)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
        
    with col_b:
        st.subheader("Classification Report")
        report = classification_report(y_eval, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, width=True)
else:
    st.error("The uploaded CSV file must contain a `target` column to compute evaluation metrics.")