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
    """ Cleans and prepares the CSV. CRASH-PROOF: Fills missing cols with 0. """
    # 1. SMART RENAMING
    df = smart_rename(df)

    # 2. SAFE TIMESTAMP (Default to 2024-01-01 if missing)
    if 'timestamp' not in df.columns:
        df['timestamp_dt'] = pd.to_datetime('2024-01-01')
    else:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
        df['timestamp_dt'] = df['timestamp_dt'].fillna(pd.to_datetime('2024-01-01'))

    # 3. SAFE SIGNUP DATE
    if 'signup_date' in df.columns:
        df['signup_date_dt'] = pd.to_datetime(df['signup_date'], errors='coerce', infer_datetime_format=True)
        df['signup_date_dt'] = df['signup_date_dt'].fillna(df['timestamp_dt'])
    else:
        df['signup_date_dt'] = df['timestamp_dt']

    # 4. FEATURE ENGINEERING
    df['account_age_days'] = (df['timestamp_dt'] - df['signup_date_dt']).dt.days
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour

    # 5. HANDLE SUCCESS
    if 'success' in df.columns:
        df['success'] = df['success'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'success'] else 0)
    else:
        df['success'] = 1 

    # 6. FILL MISSING NUMERIC COLUMNS WITH 0 (The Fix!)
    numeric_features = [
        'file_size', 'storage_limit', 'subscription_type',
        'country', 'operation', 'user_id', 'file_type', 'hour'
    ]

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0 # Create missing column with 0s

    # 7. FINAL FEATURE SELECTION
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

        # --- REAL-TIME BENCHMARKING ---
        benchmarks_data = []
        
        # Only run if enough data
        if len(df) > 10:
            y_pseudo_truth = [1 if x == -1 else 0 for x in predictions]
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_pseudo_truth, test_size=0.3, random_state=42, stratify=y_pseudo_truth
                )
                
                # A. Random Forest
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                p_rf, r_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='binary', zero_division=0)
                benchmarks_data.append({"name": "Random Forest", "Precision": round(p_rf, 2), "Recall": round(r_rf, 2), "F1": round(f1_rf, 2)})

                # B. Logistic Regression
                lr = LogisticRegression(random_state=42, max_iter=200)
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)
                p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='binary', zero_division=0)
                benchmarks_data.append({"name": "Logistic Regression", "Precision": round(p_lr, 2), "Recall": round(r_lr, 2), "F1": round(f1_lr, 2)})
            except Exception:
                pass

        # --- SUMMARY ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum())
        }

        # --- PREPARE JSON OUTPUT ---
        df_export = df.copy()
        
        # Format Timestamp
        df_export['timestamp'] = df_export['timestamp_dt'].astype(str)
        cols_to_drop = ['timestamp_dt', 'signup_date_dt']
        df_export = df_export.drop(columns=[c for c in cols_to_drop if c in df_export.columns])

        # Renaming
        pretty_names = {
            'user_id': 'User ID', 'anomaly_label': 'Status', 'anomaly_score': 'Risk Score',
            'timestamp': 'Time', 'country': 'Country', 'operation': 'Action',
            'file_type': 'File Type', 'file_size': 'Size', 'subscription_type': 'Plan',
            'account_age_days': 'Account Age', 'is_weekend': 'Weekend', 'day_of_week': 'Day',
            'storage_limit': 'Limit', 'success': 'Success', 'hour': 'Hour of Day'
        }
        df_export = df_export.rename(columns=pretty_names)

        # Mappings
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

        # Reorder
        display_id_col = 'User ID' 
        if display_id_col not in df_export.columns:
             for col in df_export.columns:
                 if 'User' in col or 'ID' in col:
                     display_id_col = col
                     break
        
        desired_order = [
            display_id_col, 'Status', 'Risk Score', 'Time', 'Country', 'Action', 
            'File Type', 'Plan', 'Size', 'Limit', 'Weekend', 'Day', 'Account Age'
        ]
        
        existing_cols = df_export.columns.tolist()
        final_order = [c for c in desired_order if c in existing_cols] + [c for c in existing_cols if c not in desired_order]
        df_export = df_export[final_order]
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