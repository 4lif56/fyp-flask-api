from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
CORS(app)

def smart_rename(df):
    """
    Scans columns and renames them to standard names if they match common keywords.
    """
    # Define synonyms for our critical columns
    column_mappings = {
        'timestamp': ['date', 'time', 'created_at', 'datetime', 'log_time', 'event_time'],
        'user_id': ['id', 'client_id', 'customer_id', 'account_id', 'username', 'user'],
        'file_size': ['size', 'bytes', 'length', 'storage_used'],
        'operation': ['action', 'activity', 'event', 'method', 'type'],
        'success': ['status', 'is_success', 'result', 'successful']
    }

    # Create a reverse mapping (Old Name -> New Name)
    rename_dict = {}
    for standard_name, synonyms in column_mappings.items():
        # If the standard name is already there, skip
        if standard_name in df.columns:
            continue
            
        # Check if any synonym exists in the user's dataframe
        for col in df.columns:
            if col.lower() in synonyms:
                rename_dict[col] = standard_name
                break # Stop after finding the first match for this column

    if rename_dict:
        print(f"Auto-renaming columns: {rename_dict}")
        df = df.rename(columns=rename_dict)
        
    return df

def preprocess_data(df):
    """
    Cleans and prepares the uploaded CSV file with maximum flexibility.
    """
    
    # 1. SMART RENAMING
    # Try to understand the user's column names
    df = smart_rename(df)

    # 2. CRITICAL CHECK: Timestamp
    # We simply cannot run time-based anomaly detection without time.
    if 'timestamp' not in df.columns:
        raise ValueError("Critical Error: No 'timestamp' or 'date' column found. Please include a time column.")

    # 3. UNIVERSAL TIME PARSING
    # 'coerce' turns garbage into NaT (Not a Time). 'infer_datetime_format' makes it fast and smart.
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
    
    # Drop rows where time was unreadable
    df = df.dropna(subset=['timestamp_dt'])

    # 4. SIGNUP DATE (Optional but good)
    if 'signup_date' in df.columns:
        df['signup_date_dt'] = pd.to_datetime(df['signup_date'], errors='coerce', infer_datetime_format=True)
        # If signup date is missing/bad, assume it's the same as the event time (account age = 0)
        df['signup_date_dt'] = df['signup_date_dt'].fillna(df['timestamp_dt'])
    else:
        # If column doesn't exist, create it locally for the math
        df['signup_date_dt'] = df['timestamp_dt']

    # 5. FEATURE ENGINEERING (The "Secret Sauce")
    # These features work regardless of what the original columns looked like
    df['account_age_days'] = (df['timestamp_dt'] - df['signup_date_dt']).dt.days
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Auto-create hour if missing
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour

    # 6. HANDLE SUCCESS COLUMN
    if 'success' in df.columns:
        # Flexible boolean parsing: handles "True", "true", 1, "Yes", etc.
        df['success'] = df['success'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'success'] else 0)
    else:
        df['success'] = 1 # Assume success if not specified

    # 7. HANDLE NUMERIC COLUMNS (Robust Cleaning)
    # These are the columns the model EXPECTS.
    # If they exist, we clean them. If not, we fill with 0.
    numeric_features = [
        'file_size', 'storage_limit', 'subscription_type',
        'country', 'operation', 'user_id', 'file_type', 'hour'
    ]

    for col in numeric_features:
        if col in df.columns:
            # Force to number, turn errors (like "5MB") into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If missing, fill with 0 so the model has *something* to read
            df[col] = 0 

    # 8. FINAL FEATURE SELECTION
    feature_columns = [
        'user_id', 'operation', 'file_type', 'file_size', 'success',
        'subscription_type', 'storage_limit', 'country', 'hour',
        'account_age_days', 'day_of_week', 'is_weekend'
    ]

    # Final cleanup: Replace any remaining NaNs (from bad parsing) with 0
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
            # Read CSV. Using 'utf-8-sig' handles BOM if created in Excel
            df_original = pd.read_csv(file, encoding='utf-8-sig') 
        except Exception as e:
            return jsonify({"error": f"Could not read CSV file: {str(e)}"}), 400
        
        # --- PREPROCESS ---
        try:
            # Pass a copy so we don't mutate the original view yet
            df, feature_columns = preprocess_data(df_original.copy())
        except ValueError as e:
            # This catches our custom "Missing Timestamp" error
            return jsonify({"error": str(e)}), 400

        if df.empty:
            return jsonify({"error": "No valid data left after preprocessing. Check your date formats."}), 400

        # --- SCALE ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])

        # --- MODEL ---
        iso = IsolationForest(contamination=0.01, random_state=42)
        predictions = iso.fit_predict(X_scaled)

        # Map results
        df['anomaly_score'] = predictions
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # --- SUMMARY ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum())
        }

        # --- PREPARE JSON OUTPUT (Smart Display) ---
        df_export = df.copy()
        
        # Convert timestamps to string for JSON
        df_export['timestamp'] = df_export['timestamp_dt'].astype(str)

        # Determine the best ID column to show first
        # We look for whatever column mapped to 'user_id' OR creates a good ID
        priority_id_cols = ['user_id', 'client_id', 'id', 'username', 'user']
        display_id_col = 'user_id' # Default
        
        for col in df_export.columns:
            if col.lower() in priority_id_cols:
                display_id_col = col
                break

        # Define column order: Result -> Score -> ID -> Time -> The Rest
        cols_priority = ['anomaly_label', 'anomaly_score', display_id_col, 'timestamp', 'operation']
        cols_rest = [c for c in df_export.columns if c not in cols_priority]
        
        # Reorder safely (only include cols that actually exist)
        final_order = [c for c in (cols_priority + cols_rest) if c in df_export.columns]
        df_export = df_export[final_order]

        # Final Clean for JSON
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