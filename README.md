# Churn Prediction Using Monthly Transactions

## Overview
This project predicts customer churn using a dataset of monthly transactions and additional customer information. The model is based on a **Random Forest Classifier** with engineered features such as rolling averages, transaction differences, and recency indicators.

## Features & Approach
- **Feature Engineering:**
  - Rolling averages, standard deviations, and max values over a time window.
  - Differences between consecutive transactions.
  - Recency-based features.
  - Plan change indicators.
  - One-hot encoding for categorical variables.
- **Data Processing:**
  - Missing values are imputed within each customer group using backward filling.
  - Data is sorted chronologically per customer.
  - Customers are split into training and test sets to prevent data leakage.
- **Model Selection:**
  - **Random Forest Classifier** is chosen for its interpretability and handling of imbalanced data.
  - **GridSearchCV** is used to optimize hyperparameters.
  - **Class weights** are used to mitigate class imbalance.

## Installation
Ensure you have Python installed and install required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### 1. Load and Process Data
Modify the file path in `FILE_PATH` before running the script.
```python
FILE_PATH = "C:/Users/USER/Downloads/churn_data.csv"
OUTPUT_PATH = "C:/Users/USER/Downloads/"
Consider also using different rolling_window other than 5
```

### 2. Feature engineering
- Calculate months since account creation
- Rolling avg/std/max for transactions
- Transactions difference from N months back
- Same for day since last transaction
- Plan type - rank the different types, check when plan changes, apply moving average on both features. Also add one-hot encoder

 
### 3. Impute
- Performed for each customer separately, to avoid data leakage. 
- used forward fill, except for plan type, where it simply replaced by 0 ("lowest" plan)

### 4. Train and Evaluate Model
- Data was splitted by customers ids - to avoid mix of same customers in train and test
- Timeline was similar for all customers, and in small range, so it was probably not critical to separate times 

## Results & Insights (what worked and also what did not work)
- **SMOTE and additional oversampling did not improve performance**, so weighted class balancing was used instead.
- The model performed best when differentiating recent transactions.
- Dimantionality reduction with mehods such as PCA did not improve the results, so were neglected (for clean code purposes, code was not added).
- Changing the model into linear, with continous explained variable, did not contribute much for better predictions.
- Applying naively LSTM on the transactions ordered by months to predict whether ANY churn happened, also seemed to be unstable (code was not added) 
l

## Future Improvements
- While performance are currently modest, There are still better than random. Additional steps - such as taking into account previous churns, as well as extracting additional data on customers, could be quite beneficia
- Experiment with deep learning models (e.g., LSTMs for sequence modeling).



