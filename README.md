# Credit Card Fraud Detection

A machine learning project that analyzes transaction data and evaluates fraud-detection classifiers with a notebook-first workflow.

## What this project demonstrates

- Data preprocessing and exploratory analysis on fraud datasets
- Supervised model training and comparison (including logistic regression and random forest)
- Precision/recall-focused evaluation for imbalanced classification
- Hyperparameter tuning using cross-validation (`GridSearchCV` with `cv=5`)

## Tech Stack

- Python
- Jupyter Notebook
- Pandas, NumPy
- scikit-learn

## Repository Contents

- `project final.ipynb` - core experiment notebook
- Supporting report/proposal/presentation files

## Dataset

- Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Notes on CV alignment

This repo clearly supports claims around model comparison and precision/recall evaluation.
SMOTE-specific implementation is not obvious in the current tracked notebook and should only be claimed if added/documented.
