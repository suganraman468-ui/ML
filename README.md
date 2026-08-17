# WILP M.Tech Machine Learning — Assignment 2

## Interactive Classification Model Evaluation Dashboard

This project implements and evaluates five supervised classification models on the **Breast Cancer Wisconsin Diagnostic dataset** and provides an interactive **Streamlit dashboard** for model selection and evaluation.

The application allows the user to upload test data, select a trained classification model, view evaluation metrics, inspect the confusion matrix, and review the classification report.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Description](#2-dataset-description)
3. [Data Preparation and Preprocessing](#3-data-preparation-and-preprocessing)
4. [Models Used](#4-models-used)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Model Performance Comparison](#6-model-performance-comparison)
7. [Model Performance Observations](#7-model-performance-observations)
8. [Logistic Regression Confusion Matrix](#8-logistic-regression-confusion-matrix)
9. [Streamlit Application](#9-streamlit-application)
10. [Application Workflow](#10-application-workflow)
11. [Repository Structure](#11-repository-structure)
12. [Technologies Used](#12-technologies-used)
13. [Installation and Local Execution](#13-installation-and-local-execution)
14. [GitHub Repository](#14-github-repository)
15. [Live Streamlit Application](#15-live-streamlit-application)
16. [BITS Virtual Lab Execution](#16-bits-virtual-lab-execution)
17. [Conclusion](#17-conclusion)

---

## 1. Problem Statement

The objective is to implement multiple machine learning classification algorithms on the same dataset, evaluate their predictive performance using standard classification metrics, and deploy the models through an interactive Streamlit web application.

**Implemented models:**

| # | Model |
|---|-------|
| 1 | Logistic Regression |
| 2 | Decision Tree Classifier |
| 3 | K-Nearest Neighbour (kNN) Classifier |
| 4 | Gaussian Naive Bayes |
| 5 | Random Forest Classifier |

**Evaluation metrics:** Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description

The project uses the **Breast Cancer Wisconsin Diagnostic dataset**, accessed through the `scikit-learn` `load_breast_cancer()` dataset loader.

| Property | Value |
|---|---|
| Total instances | 569 |
| Input features | 30 |
| Target classes | 2 |
| Classification type | Binary |
| Training instances | 455 |
| Test instances | 114 |
| Test Class 0 samples | 42 |
| Test Class 1 samples | 72 |

The 30 numerical features describe characteristics including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

The target variable is `target`:

| Value | Label |
|---|---|
| `0` | Malignant |
| `1` | Benign |

---

## 3. Data Preparation and Preprocessing

The training workflow follows these steps:

1. Load the dataset using `load_breast_cancer()`.
2. Separate features (`X`) and target (`y`).
3. Split the data into training and test sets using an 80:20 split.
4. Use `random_state=42` and stratification for reproducibility and class balance.
5. Apply `StandardScaler` for Logistic Regression and kNN.
6. Train all five classification models.
7. Save the trained models and scaler using Joblib.

> **Note:** Logistic Regression and kNN use scaled features. Decision Tree, Gaussian Naive Bayes, and Random Forest use the original feature values.

---

## 4. Models Used

| Model | Description |
|---|---|
| **Logistic Regression** | A linear binary classification model that estimates class probabilities. |
| **Decision Tree** | A tree-based model that recursively creates decision rules from the feature space. |
| **kNN** | A distance-based classifier using `n_neighbors=5`. Feature scaling is applied before training and evaluation. |
| **Gaussian Naive Bayes** | A probabilistic classifier based on Bayes' theorem and the Gaussian distribution assumption for continuous features. |
| **Random Forest** | An ensemble model that combines multiple decision trees to improve predictive performance and robustness. |

---

## 5. Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Proportion of correctly classified observations. |
| **AUC** | Ability of the model to distinguish between the two classes across thresholds. |
| **Precision** | Proportion of predicted positives that are actually positive. |
| **Recall** | Proportion of actual positives correctly identified. |
| **F1 Score** | Harmonic mean of precision and recall. |
| **MCC** | Correlation-based measure using all four confusion-matrix components. |

---

## 6. Model Performance Comparison

Results obtained by evaluating all five models on the same 114-record test dataset:

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

---

## 7. Model Performance Observations

| ML Model | Observation |
|---|---|
| **Logistic Regression** | Best overall performance. Achieved the highest Accuracy, AUC, Precision, Recall, F1 Score, and MCC. |
| **Decision Tree** | Lowest overall performance, with the lowest Accuracy and MCC among the five models. |
| **kNN** | Strong performance with 0.9561 Accuracy and 0.9655 F1 Score. Recall was 0.9722. |
| **Naive Bayes** | Moderate performance with 0.9386 Accuracy and strong AUC of 0.9878. |
| **Random Forest** | Strong performance and very high AUC of 0.9937. Matched kNN on Accuracy, Precision, Recall, F1 Score, and MCC. |

### Overall Winner

**Logistic Regression** is the overall winner for this dataset. It achieved the highest Accuracy, Precision, Recall, F1 Score, and MCC, and also recorded the highest AUC among the five models.

---

## 8. Logistic Regression Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 41 | 1 |
| **Actual 1** | 1 | 71 |

Only **2 of 114 test observations** were misclassified.

---

## 9. Streamlit Application

The application is an interactive **Classification Model Evaluation Dashboard**.

**Features:**

- CSV test dataset upload
- Classification model selection dropdown
- Automatic loading of saved models
- Feature scaling for Logistic Regression and kNN
- Prediction generation
- Evaluation metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, MCC Score
- Confusion Matrix visualization
- Classification Report

---

## 10. Application Workflow

```text
Test Dataset
     |
     v
CSV Upload / Default test_data.csv
     |
     v
Separate Features and Target
     |
     v
Feature Scaling
(Logistic Regression / kNN)
     |
     v
Select Classification Model
     |
     v
Generate Predictions
     |
     +-------------------------------+
     |                               |
     v                               v
Evaluation Metrics             Confusion Matrix
     |                               |
     +---------------+---------------+
                      |
                      v
              Classification Report
```

---

## 11. Repository Structure

```text
ML/
│
├── app.py
├── train_models.py
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit application for model selection and evaluation |
| `train_models.py` | Dataset loading, preprocessing, model training, and model saving |
| `requirements.txt` | Required Python dependencies |
| `test_data.csv` | Test dataset used by the application |
| `model/` | Saved models and feature scaler |

---

## 12. Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Visualization | Matplotlib, Seaborn |
| Web App Framework | Streamlit |

---

## 13. Installation and Local Execution

### Clone the repository

```bash
git clone https://github.com/suganraman468-ui/ML.git
cd ML
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the models

```bash
python train_models.py
```

### Run the Streamlit application

```bash
streamlit run app.py
```

Then open the Streamlit URL displayed in the terminal, normally:

```text
http://localhost:8501
```

---

## 14. GitHub Repository

🔗 [https://github.com/suganraman468-ui/ML](https://github.com/suganraman468-ui/ML)

---

## 15. Live Streamlit Application

🔗 [https://mtcukmnxvehojwgjjfzgas.streamlit.app/](https://mtcukmnxvehojwgjjfzgas.streamlit.app/)

---

## 16. BITS Virtual Lab Execution

The project was executed in the **BITS Virtual Lab** environment. The model training execution successfully generated:

- `logistic_regression.pkl`
- `decision_tree.pkl`
- `knn.pkl`
- `naive_bayes.pkl`
- `random_forest.pkl`
- `scaler.pkl`

---

## 17. Conclusion

The project demonstrates an end-to-end supervised machine learning classification workflow covering dataset loading, preprocessing, train-test splitting, model training, model persistence, evaluation, and interactive web deployment.

**Logistic Regression** achieved the strongest overall performance on the supplied test dataset:

| Metric | Score |
|---|---|
| Accuracy | 98.25% |
| AUC | 99.54% |
| Precision | 98.61% |
| Recall | 98.61% |
| F1 Score | 98.61% |
| MCC | 0.9623 |

The Streamlit dashboard provides a practical interface for selecting the trained models and reviewing their evaluation results.
