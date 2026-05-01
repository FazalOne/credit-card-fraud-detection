# Credit Card Fraud Detection

A machine learning project that analyzes transaction data and evaluates fraud-detection classifiers with a notebook-first workflow.

## What this project demonstrates

- Data preprocessing and exploratory analysis on fraud datasets
- Supervised model training and comparison (including logistic regression and random forest)
- Precision/recall-focused evaluation for imbalanced classification
- Hyperparameter tuning using cross-validation (`GridSearchCV` with `cv=5`)
- SMOTE-based oversampling on training data to improve minority-class learning
- Multi-dataset benchmarking from a single notebook driver

## Tech Stack

- Python
- Jupyter Notebook
- Pandas, NumPy
- scikit-learn

## Repository Contents

- `project final.ipynb` - core experiment notebook
- Supporting report/proposal/presentation files

## Notebook Driver Workflow

The notebook includes a driver that can run models against multiple dataset preparation modes:

- `original_data`
- `less_data`
- `less_data_smote`
- `ratio_1_to_1`

You can select datasets interactively in the notebook driver using numeric input:

- `1` -> original data
- `2` -> less data
- `3` -> less data with SMOTE
- `4` -> 1:1 ratio undersampled data

Examples:

- `1 2` runs original + less data
- `2 3 4` runs those three modes
- `all` or Enter runs all modes

If your selection includes less-data modes (`2` or `3`), the driver prompts for the number of entries to use.

The driver logs:

- model name
- best hyperparameters
- dataset mode
- train/test metrics
- train/test confusion matrices with explicit TN/FP/FN/TP

## Dataset

- Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Notes on CV alignment

This repo supports claims around model comparison, precision/recall evaluation, and SMOTE usage for imbalance handling.
