import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """
    Cleans and prepares the cleaned_data.csv file for machine learning.
    """
    # We MUST drop rows with missing timestamps because we cannot perform
    # feature engineering (like 'account_age_days') on them.
    df = df.dropna(subset=['timestamp', 'hour'])

    # 1. Convert timestamp and signup_date to datetime objects
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    except ValueError:
        df['timestamp'] = pd.to_datetime(df['timestamp']) # Try default parser

    try:
        df['signup_date'] = pd.to_datetime(df['signup_date'], format='%d/%m/%Y')
    except ValueError:
        df['signup_date'] = pd.to_datetime(df['signup_date']) # Try default parser

    # 2. Feature Engineering
    df['account_age_days'] = (df['timestamp'] - df['signup_date']).dt.days
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['success'] = df['success'].apply(lambda x: 1 if x else 0)

    # 4. Define features
    feature_columns = [
        'user_id', 'operation', 'file_type', 'file_size', 'success',
        'subscription_type', 'storage_limit', 'country', 'hour',
        'account_age_days', 'day_of_week', 'is_weekend'
    ]
    
    final_df = df[feature_columns].copy()
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.fillna(0, inplace=True)
    
    return final_df, feature_columns

# --- Main Script ---
print("Running comparative evaluation for Chapter 4...")
print("Loading and preprocessing data...")
try:
    df = pd.read_csv('cleaned_data.csv')
except FileNotFoundError:
    print("Error: cleaned_data.csv not found.")
    exit() 

# Preprocess the data
X, feature_columns = preprocess_data(df.copy())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data preprocessed. {len(X)} rows ready for modelling.")
print("-" * 30)

# --- STEP 1: ISOLATION FOREST (GET LABELS) ---
print("Running Model 1: Isolation Forest (Unsupervised)...")
contamination_rate = 0.01
iforest = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
iforest.fit(X_scaled)
y_pred_iforest = iforest.predict(X_scaled)

# Create our "answer key" (y)
y = [1 if pred == -1 else 0 for pred in y_pred_iforest]
anomalies_found = sum(y)
print(f"Isolation Forest found {anomalies_found} anomalies (out of {len(X)} rows).")
print("-" * 30)

# --- STEP 2: SPLIT DATA ---
print("Splitting data for supervised model comparison...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print("-" * 30)

# --- STEP 3: RANDOM FOREST (SUPERVISED) ---
print("Running Model 2: Random Forest (Supervised)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- FINAL EVALUATION RESULTS ---")
print("\nRandom Forest Evaluation (Model 2):")
# We only care about the 'Anomaly (1)' row
print(classification_report(y_test, y_pred_rf, target_names=['Normal (0)', 'Anomaly (1)']))

# --- STEP 4: LOGISTIC REGRESSION (SUPERVISED) ---
print("Running Model 3: Logistic Regression (Supervised)...")
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Evaluation (Model 3):")
# We only care about the 'Anomaly (1)' row
print(classification_report(y_test, y_pred_lr, target_names=['Normal (0)', 'Anomaly (1)']))
print("---------------------------------")