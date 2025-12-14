# Fake Banknote Detection using K-Nearest Neighbors

## Overview
This project implements a supervised machine learning model to classify banknotes as genuine or counterfeit based on physical measurement features. The objective is to minimize the risk of falsely identifying counterfeit bills as genuine, a scenario with higher real-world cost.

## Dataset
The dataset consists of 1,500 observations and 6 numerical features describing dimensional characteristics of banknotes. The target variable indicates whether a bill is genuine or counterfeit. The data contains a small number of missing values and an imbalanced class distribution.

## Approach
- Exploratory Data Analysis (EDA) to understand feature distributions and class imbalance
- Data preprocessing with missing-value imputation using **KNNImputer**
- Feature selection using **SelectKBest**
- Model training with **K-Nearest Neighbors (KNN)**
- Hyperparameter tuning using **GridSearchCV**
- Stratified cross-validation to preserve class balance
- Evaluation using accuracy, precision, recall, and confusion matrix

All preprocessing and modeling steps were implemented using a reproducible **scikit-learn Pipeline** to prevent data leakage.

## Results
- Cross-validated accuracy: ~99%
- Test-set precision: ~99%
- The model demonstrates strong performance in correctly identifying counterfeit bills while minimizing false positives.

## Technologies Used
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Key Learnings
- Importance of handling class imbalance in classification problems
- Benefits of pipelines for reproducibility and leakage prevention
- Tradeoffs between evaluation metrics depending on business context

## Project Structure
- `FakeBillsWithKNN.ipynb` — main analysis and modeling notebook
- `fake_bills.csv` — dataset
