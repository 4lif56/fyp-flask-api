from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

app = Flask(__name__)
CORS(app)

def preprocess_data(df):
    """
    Cleans and prepares ANY uploaded CSV file for anomaly detection.
    Handles DD/MM/YYYY and other mixed date formats safely.
    """
    # --- 1. Ensure timestamp exists ---
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")
    df = df.dropna(subset=['timestamp'])

    # --- 2. Parse timestamp robustly ---
    df['timestamp_dt'] = pd.to_datetime(
        df['timestamp'],
        errors='coerce',  # invalid dates become NaT
        dayfirst=True      # DD/MM/YYYY format
    )
    df = df.dropna(subset=['timestamp_dt'])

    # --- 3. Signup date ---
    if 'signup_date' in df.columns:
        df['signup_date_dt'] = pd.to_datetime(
            df['signup_date'],
            errors='coerce',
            dayfirst=True
        )
        df['signup_date_dt'] = df['signup_date_dt'].fillna(df['timestamp_dt'])
    else:
        df['signup_date_dt'] = df['timestamp_dt']

    # --- 4. Feature engineering ---
    df['account_age_days'] = (df['timestamp_dt'] - df['signup_date_dt']).dt.days
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # --- 5. Hour column ---
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour

    # --- 6. Convert success to numeric ---
    if 'success' in df.columns:
        df['success'] = df['success'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    else:
        df['success'] = 1

    # --- 7. Force numeric columns ---
    numeric_cols = [
        'file_size', 'storage_limit', 'subscription_type',
        'country', 'operation', 'user_id', 'file_type', 'hour'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # --- 8. Final features ---
    feature_columns = [
        'user_id', 'operation', 'file_type', 'file_size', 'success',
        'subscription_type', 'storage_limit', 'country', 'hour',
        'account_age_days', 'day_of_week', 'is_weekend'
    ]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df[feature_columns] = df[feature_columns].fillna(0)

    return df, feature_columns


@app.route('/detect', methods=['POST'])
def detect():
    start_time = time.time()
    # --- Initialize summary early to prevent missing data errors ---
    summary = {"total_rows": 0, "anomalies": 0, "duration": 0}
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded", "summary": summary}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected", "summary": summary}), 400

        # --- Load CSV ---
        try:
            df_original = pd.read_csv(file, sep=None, engine='python')
        except Exception:
            file.seek(0)
            df_original = pd.read_csv(file, sep='\t')

        if df_original.empty:
            return jsonify({"summary": summary, "results": []})

        # --- Preprocess ---
        df, feature_columns = preprocess_data(df_original.copy())
        if df.empty:
            return jsonify({"summary": summary, "results": []})

        # --- Scale features ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])

        # --- Isolation Forest ---
        iso = IsolationForest(contamination=0.01, random_state=42)
        predictions = iso.fit_predict(X_scaled)
        df['anomaly_score'] = predictions
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # --- Summary ---
        total_rows = len(df)
        anomalies = int((df['anomaly_label'] == 'Anomaly').sum())
        duration = round(time.time() - start_time, 3)

        summary = {
            "total_rows": total_rows,
            "anomalies": anomalies,
            "duration": duration
        }

        # --- Prepare results ---
        df_export = df.copy()
        df_export['timestamp'] = df_export['timestamp_dt'].astype(str)
        df_export = df_export.drop(columns=['timestamp_dt', 'signup_date_dt'], errors='ignore')
        df_export = df_export.replace([np.inf, -np.inf], "").fillna("")

        return jsonify({
            "summary": summary,
            "results": df_export.to_dict(orient="records")
        })

    except Exception as e:
        print("Server Error:", str(e))
        return jsonify({"error": f"Server error: {str(e)}", "summary": summary}), 500
