🧬 Breast Cancer Classification – Logistic Regression & KNN
📌 Project Overview

This project applies Machine Learning to classify breast cancer tumors as Malignant (cancerous) or Benign (non-cancerous) using the Breast Cancer Wisconsin Dataset.
The main objective is to explore the dataset, preprocess features, and build predictive models for early diagnosis.

🔎 Workflow

Data Loading & EDA

Used sklearn.datasets.load_breast_cancer

Checked for missing values (none found)

Visualized diagnosis distribution (Malignant vs Benign)

Data Preprocessing

Converted diagnosis to binary (1 = Malignant, 0 = Benign)

Removed highly correlated features (reduced from 32 → 23 features)

Standardized features with StandardScaler

Model Training

Logistic Regression

Train Accuracy: 98.9%

Test Accuracy: 96.5%

Only 4 misclassifications on test set

K-Nearest Neighbors (KNN)

Trained with default K=5

Performance compared with Logistic Regression

Evaluation

Confusion Matrix, Classification Report

Accuracy scores for both models

📊 Key Results

Logistic Regression performed better with ~96.5% test accuracy

KNN also worked well, but tuning K is needed for optimization

Very low number of misclassifications, showing models are effective

🚀 Next Improvements

Hyperparameter tuning for KNN (optimal K search)

Add SVM, Random Forest, Gradient Boosting for comparison

Evaluate with ROC-AUC curves & Precision-Recall

Interpret features with Logistic Regression coefficients

🛠️ Tech Stack

Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn
