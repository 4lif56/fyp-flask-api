import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# ==========================================
# 1. THE SHARED LOGIC (Copied from app.py)
# ==========================================
def preprocess_data(df):
    """ Cleans and prepares the CSV exactly like the App does. """
    
    # 1. SMART RENAME
    rename_map = {
        'timestamp': ['date', 'time', 'created_at', 'datetime'],
        'user_id': ['id', 'client_id', 'customer_id', 'account_id', 'username'],
        'file_size': ['size', 'bytes', 'length'],
        'operation': ['action', 'activity', 'event', 'type'],
        'success': ['status', 'is_success', 'result'],
        'country': ['region', 'location', 'geo'],
        'file_type': ['file_format', 'extension'],
        'subscription_type': ['plan', 'tier']
    }
    
    curr_cols = set(df.columns)
    cols_to_rename = {}
    for std, alts in rename_map.items():
        if std not in curr_cols:
            for alt in alts:
                match = next((c for c in curr_cols if c.lower() == alt), None)
                if match:
                    cols_to_rename[match] = std
                    break
    
    if cols_to_rename:
        df.rename(columns=cols_to_rename, inplace=True)

    # 2. FILL MISSING & FEATURE ENGINEERING
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    else:
        df['timestamp_dt'] = pd.Timestamp('2024-01-01')

    # Account Age
    if 'signup_date' in df.columns:
        signup_dt = pd.to_datetime(df['signup_date'], errors='coerce').fillna(df['timestamp_dt'])
        df['account_age_days'] = (df['timestamp_dt'] - signup_dt).dt.days.fillna(0).astype('int16')
    else:
        df['account_age_days'] = 0

    # Date Features
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')

    # Success Flag
    if 'success' in df.columns:
        df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes', 'success']).astype('int8')
    else:
        df['success'] = 1

    # Ensure all required numeric columns exist
    req_cols = ['file_size', 'storage_limit', 'subscription_type', 'country', 'operation', 'user_id', 'file_type', 'hour']
    for c in req_cols:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # The Final List of 12 Features (MUST MATCH APP.PY)
    features = [
        'user_id', 'operation', 'file_type', 'file_size', 'success',
        'subscription_type', 'storage_limit', 'country', 'hour',
        'account_age_days', 'day_of_week', 'is_weekend'
    ]
    
    df[features] = df[features].fillna(0)
    
    return df, features

# ==========================================
# 2. TRAINING SCRIPT
# ==========================================
print("LOADING data...")
df_raw = pd.read_csv('cleaned_data.csv')

print("PRE-PROCESSING data (Adding 4 extra features)...")
df_clean, feature_cols = preprocess_data(df_raw)

print(f"Features used for training ({len(feature_cols)}): {feature_cols}")

print("SCALING data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[feature_cols])

print("TRAINING model... (This is the brain)")
model = IsolationForest(n_estimators=100, contamination=0.05, n_jobs=-1, random_state=42)
model.fit(X_scaled)

print("SAVING model and scaler...")
joblib.dump(model, 'model.joblib', compress=3)
joblib.dump(scaler, 'scaler.joblib', compress=3)

print("✅ DONE! The model now knows about 'success' and 'weekend'.")