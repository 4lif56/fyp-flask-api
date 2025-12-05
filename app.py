from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
CORS(app)

# --- FIX 1: DISABLE ALPHABETICAL SORTING ---
# This ensures the JSON stays in the exact order we define (User ID first)
app.json.sort_keys = False 

def smart_rename(df):
    """
    Scans columns and renames them to standard names if they match common keywords.
    """
    column_mappings = {
        'timestamp': ['date', 'time', 'created_at', 'datetime', 'log_time', 'event_time'],
        'user_id': ['id', 'client_id', 'customer_id', 'account_id', 'username', 'user'],
        'file_size': ['size', 'bytes', 'length', 'storage_used'],
        'operation': ['action', 'activity', 'event', 'method', 'type'],
        'success': ['status', 'is_success', 'result', 'successful'],
        'country': ['region', 'location', 'geo', 'zone'],
        'file_type': ['file_format', 'extension', 'type'],
        'subscription_type': ['plan', 'tier', 'subscription']
    }

    rename_dict = {}
    for standard_name, synonyms in column_mappings.items():
        if standard_name in df.columns:
            continue
        for col in df.columns:
            if col.lower() in synonyms:
                rename_dict[col] = standard_name
                break

    if rename_dict:
        print(f"Auto-renaming columns: {rename_dict}")
        df = df.rename(columns=rename_dict)
        
    return df

def preprocess_data(df):
    """
    Cleans and prepares the uploaded CSV file.
    """
    # 1. SMART RENAMING
    df = smart_rename(df)

    # 2. CRITICAL CHECK: Timestamp
    if 'timestamp' not in df.columns:
        raise ValueError("Critical Error: No 'timestamp' or 'date' column found.")

    # 3. UNIVERSAL TIME PARSING
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=['timestamp_dt'])

    # 4. SIGNUP DATE
    if 'signup_date' in df.columns:
        df['signup_date_dt'] = pd.to_datetime(df['signup_date'], errors='coerce', infer_datetime_format=True)
        df['signup_date_dt'] = df['signup_date_dt'].fillna(df['timestamp_dt'])
    else:
        df['signup_date_dt'] = df['timestamp_dt']

    # 5. FEATURE ENGINEERING
    df['account_age_days'] = (df['timestamp_dt'] - df['signup_date_dt']).dt.days
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour

    # 6. HANDLE SUCCESS
    if 'success' in df.columns:
        df['success'] = df['success'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'success'] else 0)
    else:
        df['success'] = 1 

    # 7. HANDLE NUMERIC COLUMNS
    numeric_features = [
        'file_size', 'storage_limit', 'subscription_type',
        'country', 'operation', 'user_id', 'file_type', 'hour'
    ]

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0 

    # 8. FINAL FEATURE SELECTION
    feature_columns = [
        'user_id', 'operation', 'file_type', 'file_size', 'success',
        'subscription_type', 'storage_limit', 'country', 'hour',
        'account_age_days', 'day_of_week', 'is_weekend'
    ]

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
            df_original = pd.read_csv(file, encoding='utf-8-sig') 
        except Exception as e:
            return jsonify({"error": f"Could not read CSV file: {str(e)}"}), 400
        
        # --- PREPROCESS ---
        try:
            df, feature_columns = preprocess_data(df_original.copy())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if df.empty:
            return jsonify({"error": "No valid data left after preprocessing."}), 400

        # --- SCALE & MODEL ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])

        iso = IsolationForest(contamination=0.01, random_state=42)
        predictions = iso.fit_predict(X_scaled)

        df['anomaly_score'] = predictions
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # --- SUMMARY ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum())
        }

        # --- PREPARE JSON OUTPUT ---
        df_export = df.copy()
        
        # 1. TRANSLATE CODES BACK TO NAMES
        mappings = {
            'country': {0: 'DE', 1: 'FR', 2: 'GB', 3: 'PL', 4: 'UA', 5: 'US'},
            'operation': {0: 'delete', 1: 'download', 2: 'modify', 3: 'upload'},
            'file_type': {0: 'archive', 1: 'document', 2: 'photo', 3: 'video'},
            'subscription_type': {0: 'business', 1: 'free', 2: 'premium'}
        }

        for col, mapping_dict in mappings.items():
            if col in df_export.columns:
                df_export[col] = df_export[col].astype(int).map(mapping_dict).fillna(df_export[col])

        # 2. Format Timestamp
        df_export['timestamp'] = df_export['timestamp_dt'].astype(str)

        # 3. FIX: Drop helper columns so they don't show in UI
        cols_to_drop = ['timestamp_dt', 'signup_date_dt']
        df_export = df_export.drop(columns=[c for c in cols_to_drop if c in df_export.columns])

        # 4. Smart Column Reordering
        priority_id_cols = ['user_id', 'client_id', 'id', 'username', 'user']
        display_id_col = 'user_id'
        for col in df_export.columns:
            if col.lower() in priority_id_cols:
                display_id_col = col
                break

        # Define STRICT order: ID -> Result -> Time -> Details
        cols_priority = [display_id_col, 'anomaly_label', 'anomaly_score', 'timestamp', 'country', 'operation', 'file_type', 'subscription_type']
        cols_rest = [c for c in df_export.columns if c not in cols_priority]
        final_order = [c for c in (cols_priority + cols_rest) if c in df_export.columns]
        df_export = df_export[final_order]

        # 5. Final Clean
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