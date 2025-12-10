from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

app = Flask(__name__)
CORS(app)

app.json.sort_keys = False 

def smart_rename(df):
    """ Scans columns and renames them to standard names. """
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
    DYNAMIC PREPROCESSING:
    Only processes columns that actually exist. Does not invent fake data.
    """
    # 1. SMART RENAMING
    df = smart_rename(df)

    # 2. FEATURE ENGINEERING (Only if columns exist)
    
    # Time-based features
    if 'timestamp' in df.columns:
        # Convert to datetime
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
        # We drop rows where timestamp failed to parse, ONLY IF timestamp was supposed to be there
        df = df.dropna(subset=['timestamp_dt'])
        
        # Extract features
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp_dt'].dt.hour
        
        df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Account Age (Needs both timestamp and signup_date)
        if 'signup_date' in df.columns:
            df['signup_date_dt'] = pd.to_datetime(df['signup_date'], errors='coerce', infer_datetime_format=True)
            # Fill missing signup dates with timestamp (age=0) to save the row
            df['signup_date_dt'] = df['signup_date_dt'].fillna(df['timestamp_dt'])
            df['account_age_days'] = (df['timestamp_dt'] - df['signup_date_dt']).dt.days

    # 3. HANDLE SUCCESS (Convert True/False to 1/0 if exists)
    if 'success' in df.columns:
        df['success'] = df['success'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'success'] else 0)

    # 4. SELECT VALID FEATURES DYNAMICALLY
    # List of all *possible* numeric features we know how to handle
    possible_features = [
        'file_size', 'storage_limit', 'subscription_type',
        'country', 'operation', 'file_type', 'hour', 
        'account_age_days', 'day_of_week', 'is_weekend', 'success'
    ]

    # Filter to keep ONLY the ones present in this specific file
    existing_numeric_cols = [col for col in possible_features if col in df.columns]

    # Convert them to numeric (coercing errors to NaN, then filling with 0)
    # This handles cases where a user uploads "USA" instead of "5" - it treats it as 0 safely.
    for col in existing_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Always keep User ID for display, even if not used in training
    if 'user_id' not in df.columns:
        # Generate fake IDs (1, 2, 3...) just for the table view
        df['user_id'] = range(1, len(df) + 1)

    return df, existing_numeric_cols

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
        
        # --- PREPROCESS (Dynamic) ---
        try:
            df, feature_columns = preprocess_data(df_original.copy())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if df.empty:
            return jsonify({"error": "No valid data left after processing."}), 400
            
        if not feature_columns:
            return jsonify({"error": "Could not find any usable numeric columns for analysis."}), 400

        # --- SCALE & MODEL ---
        # We train ONLY on the columns found in this file
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])

        iso = IsolationForest(contamination=0.01, random_state=42)
        predictions = iso.fit_predict(X_scaled)

        df['anomaly_score'] = predictions
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # --- REAL-TIME BENCHMARKING ---
        benchmarks_data = []
        # Only benchmark if we have enough data rows
        if len(df) > 15:
            y_pseudo_truth = [1 if x == -1 else 0 for x in predictions]
            try:
                # We need to stratify, but if there are too few anomalies (e.g. 1), stratification fails.
                # So we wrap in try/except.
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_pseudo_truth, test_size=0.3, random_state=42, stratify=y_pseudo_truth
                )
                
                # A. Random Forest
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='binary', zero_division=0)
                benchmarks_data.append({"name": "Random Forest", "Precision": round(prec_rf, 2), "Recall": round(rec_rf, 2), "F1": round(f1_rf, 2)})

                # B. Logistic Regression
                lr = LogisticRegression(random_state=42, max_iter=200)
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)
                prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='binary', zero_division=0)
                benchmarks_data.append({"name": "Logistic Regression", "Precision": round(prec_lr, 2), "Recall": round(rec_lr, 2), "F1": round(f1_lr, 2)})
            except Exception as e:
                print(f"Benchmark skipped: Not enough variance or anomalies. {e}")
                pass

        # --- SUMMARY ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum())
        }

        # --- PREPARE JSON OUTPUT ---
        df_export = df.copy()
        
        # Format Timestamp (Only if it exists)
        if 'timestamp_dt' in df_export.columns:
            df_export['timestamp'] = df_export['timestamp_dt'].astype(str)
            df_export = df_export.drop(columns=['timestamp_dt'])
        
        if 'signup_date_dt' in df_export.columns:
            df_export = df_export.drop(columns=['signup_date_dt'])

        # Renaming & Mapping (Dynamic)
        pretty_names = {
            'user_id': 'User ID', 'anomaly_label': 'Status', 'anomaly_score': 'Risk Score',
            'timestamp': 'Time', 'country': 'Country', 'operation': 'Action',
            'file_type': 'File Type', 'file_size': 'Size', 'subscription_type': 'Plan',
            'account_age_days': 'Account Age', 'is_weekend': 'Weekend', 'day_of_week': 'Day',
            'storage_limit': 'Limit', 'success': 'Success', 'hour': 'Hour of Day'
        }
        df_export = df_export.rename(columns=pretty_names)

        mappings = {
            'Country': {0: 'DE', 1: 'FR', 2: 'GB', 3: 'PL', 4: 'UA', 5: 'US'},
            'Action': {0: 'delete', 1: 'download', 2: 'modify', 3: 'upload'},
            'File Type': {0: 'archive', 1: 'document', 2: 'photo', 3: 'video'},
            'Plan': {0: 'business', 1: 'free', 2: 'premium'},
            'Weekend': {0: 'No', 1: 'Yes'},
            'Success': {1: 'Success', 0: 'Failed'} 
        }

        for col, mapping_dict in mappings.items():
            if col in df_export.columns:
                if pd.api.types.is_numeric_dtype(df_export[col]):
                    df_export[col] = df_export[col].astype(int).map(mapping_dict).fillna(df_export[col])

        # Column Reordering
        display_id_col = 'User ID' 
        # Find the ID column if renamed
        if display_id_col not in df_export.columns:
             for col in df_export.columns:
                 if 'User' in col or 'ID' in col:
                     display_id_col = col
                     break
        
        # Build desired order based on what ACTUALLY exists
        base_order = [display_id_col, 'Status', 'Risk Score', 'Time', 'Country', 'Action', 'File Type', 'Plan']
        # Filter desired_order to only include columns that are actually in df_export
        final_order = [c for c in base_order if c in df_export.columns] 
        # Add any remaining columns
        remaining_cols = [c for c in df_export.columns if c not in final_order]
        
        df_export = df_export[final_order + remaining_cols]
        df_export = df_export.replace([np.inf, -np.inf], "").fillna("")

        return jsonify({
            "summary": summary,
            "benchmarks": benchmarks_data,
            "results": df_export.to_dict(orient="records")
        })

    except Exception as e:
        print("Server Error:", str(e))
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)