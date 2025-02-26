import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score , confusion_matrix , precision_score, recall_score
import logging
import json
import os
import pickle


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# read and parse date from csv 
def load_data(filepath):
    """Loads and preprocesses the dataset."""
    logging.info("Loading dataset...")
    df = pd.read_csv(filepath, parse_dates=["date", "issuing_date"])
    df.sort_values(by=["customer_id", "date"], inplace=True)
    return df


# Features generation - main effects came from differentiating with previous months
def feature_engineering(df, rolling_window=3):
    """Performs feature engineering on the dataset."""
    logging.info("Performing feature engineering...")

    # Months since account creation
    df["months_since_issuing"] = (df["date"] - df["issuing_date"]).dt.days / 30

    # Rolling avg/std
    df["rolling_avg"] = df.groupby("customer_id")["transaction_amount"].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
    df["rolling_std"] = df.groupby("customer_id")["transaction_amount"].transform(lambda x: x.rolling(rolling_window, min_periods=1).std())
    df["rolling_max"] = df.groupby("customer_id")["transaction_amount"].transform(lambda x: x.rolling(rolling_window, min_periods=1).max())

    # Transaction diff - log diff can provide ratio, but do not contribute much
    #df['transaction_amount_log'] = df['transaction_amount'].apply(np.log10)


    for i in range(1, 4):  # Generate diff(1), diff(2), diff(3)
        df[f"transaction_diff_{i}"] = df.groupby("customer_id")["transaction_amount"].diff(i)
        #df[f"transaction_diff_log_{i}"] = df.groupby("customer_id")["transaction_amount_log"].diff(i).apply(np.exp)

    # Recency Feature
    for i in range(1, 4):
        df[f"days_since_last_tx_diff{i}"] = df.groupby("customer_id")["date"].diff(i).dt.days.fillna(0)
 
    # Plan Change Indicator
    df['plan_type'] = df.groupby("customer_id")['plan_type'].transform(lambda group: group.bfill()) # Impute missing val
    dict_plan = {'Basic':0,'Standard':1,'Premium':2} # rank plan_type, based on assumption of their importance
    df['plan_type'] = df['plan_type'].map(dict_plan) 

    df["plan_changed"] = df.groupby("customer_id")["plan_type"].apply(lambda x: x != x.shift()).astype(int).values
    df["plan_changed_mean"] = df.groupby("customer_id")["plan_changed"].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())

    # Try also treat plan_type as categorical One-Hot Encoding for categorical features (not crucial)
    df = pd.get_dummies(df, columns=["plan_type"], drop_first=True)

    return df


# Missing values - impute for each customer id, to avoid data likage 
def impute_missing_values(df):
    """Fills missing values using backward fill within each customer."""
    logging.info("Imputing missing values...")
    df_imputed = df.groupby("customer_id").apply(lambda group: group.bfill()).reset_index(drop=True)
    
    # Convert bool to int 
    bool_cols = df_imputed.select_dtypes(include=[bool]).columns
    df_imputed[bool_cols] = df_imputed[bool_cols].astype(int)
    
    return df_imputed


# Split and apply model - make sure to split by customer id, to avoid mix of customuer's "timeline"
def split_data(df, test_size=0.2, random_state=42):
    """Splits data into train and test sets without mixing the same customer in both sets."""
    logging.info("Splitting data into train and test sets...")

    unique_customers = df["customer_id"].unique()
    train_customers, test_customers = train_test_split(unique_customers, test_size=test_size, random_state=random_state)

    train_df = df[df["customer_id"].isin(train_customers)].drop(["customer_id", "issuing_date", "date"], axis=1)
    test_df = df[df["customer_id"].isin(test_customers)].drop(["customer_id", "issuing_date", "date"], axis=1)

    return train_df, test_df


# Tune parameters for model - Random forest was chosen due to it's relative simplicity and eplxainability of features
def tune_hyperparameters_random_forest(X_train, y_train):
    """Performs hyperparameter tuning using GridSearchCV."""
    logging.info("Tuning hyperparameters...")

    param_grid = {
        "n_estimators": [20, 50, 75],   # Number of trees in the forest
        "max_depth": [3, 5,7 ],          # Maximum depth of each tree
    }
    # Since classes are unbalanced, use class weight
    model = RandomForestClassifier(class_weight='balanced',random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best accuracy score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def send_results_to_json(y_pred, y_test , y_proba, output_path):
        
    os.makedirs(output_path , exist_ok=True)
    # Compute evaluation metrics
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

    # Save metrics to a JSON file
    metrics_filename = "churn_model_results.json"
    with open(os.path.join(output_path, metrics_filename), "w") as f:
        json.dump(metrics, f)
    logging.info(f"Saved results in {output_path}")

def save_model_as_pickle(model , output_path):
    # Save a model
    with open(os.path.join(output_path, 'model.pkl'), 'wb') as file:
        pickle.dump(model, file)

# main
if __name__ == "__main__":
    # Filepath to dataset (adjust as needed)
    FILE_PATH = "C:/Users/USER/Downloads/churn_data (3).csv"
    OUTPUT_PATH = "C:/Users/USER/Downloads/"

    # Load and process data
    df = load_data(FILE_PATH)
    df_processed = feature_engineering(df, rolling_window=5)
    df_fin = impute_missing_values(df_processed)

    # Split data
    train_df, test_df = split_data(df_fin, test_size=0.2)

    # Prepare train/test sets
    X_train, y_train = train_df.drop(columns=["churn"]), train_df["churn"]
    X_test, y_test = test_df.drop(columns=["churn"]), test_df["churn"]

    # Here I tried to over-sample the training set with SMOTE/over-sampler, However, it did not provide any added value
    # Additional attempts for compressing with PCA and trying to predict sum/mean of churn per customer as regression model - also did not work here
    # Hence, just use weighted class in the model itself
    
    # Train best model using GridSearchCV
    best_model = tune_hyperparameters_random_forest(X_train, y_train)

    # Predictions
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    send_results_to_json(y_pred, y_test , y_proba, output_path = OUTPUT_PATH)
    
    save_model_as_pickle(best_model , output_path=OUTPUT_PATH)

    logging.info(f"ROC AUC Score: { roc_auc_score(y_test, y_proba)}")

    logging.info("Model training complete. Predictions generated.")

    print("Classification Report:\n", classification_report(y_test, y_pred))

    print("Confusion matrix :\n", confusion_matrix(y_test, y_pred))
