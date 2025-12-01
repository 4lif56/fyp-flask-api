from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
CORS(app)

def preprocess_data(df):
    """
    Cleans and prepares the uploaded CSV file for machine learning.
    """
    # 1. Ensure timestamp exists
    if 'timestamp' not in df.columns:
        # If no timestamp, we can't do time-based anomaly detection
        # You might want to handle this differently, but for now:
        if 'time' in df.columns:
             df['timestamp'] = df['time'] # Try to recover
        else:
             return df, [] # Fail gracefully

    # Drop only if timestamp itself is missing
    df = df.dropna(subset=['timestamp'])

    # 2. UNIVERSAL timestamp parsing
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['timestamp_dt'])

    # 3. Signup date (optional)
    if 'signup_date' in df.columns:
        df['signup_date_dt'] = pd.to_datetime(df['signup_date'], errors='coerce', dayfirst=True)
        df['signup_date_dt'] = df['signup_date_dt'].fillna(df['timestamp_dt'])
    else:
        df['signup_date_dt'] = df['timestamp_dt']

    # 4. Feature Engineering
    df['account_age_days'] = (df['timestamp_dt'] - df['signup_date_dt']).dt.days
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 5. Auto-create `hour` column if missing
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour

    # 6. Convert 'success' to numeric
    if 'success' in df.columns:
        df['success'] = df['success'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    else:
        df['success'] = 1 

    # 7. Force-clean numeric columns
    numeric_cols = [
        'file_size', 'storage_limit', 'subscription_type',
        'country', 'operation', 'user_id', 'file_type', 'hour'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0 

    # 8. Final feature list
    feature_columns = [
        'user_id', 'operation', 'file_type', 'file_size', 'success',
        'subscription_type', 'storage_limit', 'country', 'hour',
        'account_age_days', 'day_of_week', 'is_weekend'
    ]

    # Missing columns → auto-fill
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Replace NaN with 0 for ML
    df[feature_columns] = df[feature_columns].fillna(0)

    return df, feature_columns


@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        try:
            df_original = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Could not read CSV file: {str(e)}"}), 400
        
        # --- PREPROCESS ---
        df, feature_columns = preprocess_data(df_original.copy())

        if df.empty:
            return jsonify({"error": "No valid data left after preprocessing."}), 400

        # --- SCALE ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])

        # --- MODEL ---
        iso = IsolationForest(contamination=0.01, random_state=42)
        predictions = iso.fit_predict(X_scaled)

        df['anomaly_score'] = predictions
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # --- SUMMARY ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum())
        }

        # --- Prepare JSON output ---
        df_export = df.copy()
        df_export['timestamp'] = df_export['timestamp_dt'].astype(str)

        # *** REORDER COLUMNS HERE ***
        # This puts the most important info on the LEFT
        column_order = [
            'anomaly_label',   # 1. Result
            'anomaly_score',   # 2. Score
            'user_id',         # 3. Who?
            'operation',       # 4. What?
            'timestamp',       # 5. When?
            'file_size',       # 6. How big?
            'country',         
            'file_type',       
            'storage_limit',
            'subscription_type',
            'success',
            'account_age_days',
            'is_weekend',
            'day_of_week',
            'hour'
        ]
        
        # Keep only columns that exist, in the correct order
        existing_cols = [c for c in column_order if c in df_export.columns]
        df_export = df_export[existing_cols]

        # Clean up JSON
        df_export = df_export.replace([np.inf, -np.inf], "").fillna("")

        return jsonify({
            "summary": summary,
            "results": df_export.to_dict(orient="records")
        })

    except Exception as e:
        print("Server Error:", str(e))
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)