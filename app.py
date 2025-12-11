from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import gc  # Garbage Collector interface

app = Flask(__name__)
CORS(app)

app.json.sort_keys = False 

def preprocess_data(df):
    """ Cleans and prepares the CSV in-place to save memory. """
    
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

    # 2. OPTIMIZED TYPE CONVERSION
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            continue
        if 'int' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif 'float' in str(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')

    # 3. FILL MISSING & FEATURE ENGINEERING
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    else:
        df['timestamp_dt'] = pd.Timestamp('2024-01-01')

    if 'signup_date' in df.columns:
        signup_dt = pd.to_datetime(df['signup_date'], errors='coerce').fillna(df['timestamp_dt'])
        df['account_age_days'] = (df['timestamp_dt'] - signup_dt).dt.days.astype('int16')
    else:
        df['account_age_days'] = 0

    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')

    if 'success' in df.columns:
        df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes', 'success']).astype('int8')
    else:
        df['success'] = 1

    req_cols = ['file_size', 'storage_limit', 'subscription_type', 'country', 'operation', 'user_id', 'file_type']
    for c in req_cols:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    features = [
        'user_id', 'operation', 'file_type', 'file_size', 'success',
        'subscription_type', 'storage_limit', 'country', 'hour',
        'account_age_days', 'day_of_week', 'is_weekend'
    ]
    return df, features

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({"error": "No file selected"}), 400

        try:
            df_original = pd.read_csv(file, encoding='utf-8-sig')
        except Exception as e:
            return jsonify({"error": f"CSV Error: {str(e)}"}), 400

        try:
            df, feature_cols = preprocess_data(df_original)
        except Exception as e:
            return jsonify({"error": f"Processing Error: {str(e)}"}), 400

        if df.empty: return jsonify({"error": "Empty dataset"}), 400

        # --- MEMORY OPTIMIZATION ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols]).astype(np.float32)

        # --- MODEL 1: ISOLATION FOREST ---
        iso = IsolationForest(contamination=0.01, n_estimators=100, n_jobs=1, random_state=42)
        predictions = iso.fit_predict(X_scaled)

        df['anomaly_score'] = predictions
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # --- MODEL 2: BENCHMARKS ---
        benchmarks_data = []
        if len(df) > 10:
            y_truth = np.where(predictions == -1, 1, 0)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_truth, test_size=0.3, random_state=42, stratify=y_truth
            )

            # Random Forest
            rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=1, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            benchmarks_data.append({"name": "Random Forest", "Precision": round(p, 2), "Recall": round(r, 2), "F1": round(f1, 2)})
            del rf
            gc.collect()

            # Logistic Regression
            lr = LogisticRegression(max_iter=200, solver='liblinear', random_state=42)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            benchmarks_data.append({"name": "Logistic Regression", "Precision": round(p, 2), "Recall": round(r, 2), "F1": round(f1, 2)})
            del lr
            gc.collect()

        # --- NEW: TIMELINE AGGREGATION (With Fixed Indentation) ---
        df['dt_temp'] = df['timestamp_dt']
        
        # Group by Date and Label
        timeline_groups = df.groupby([df['dt_temp'].dt.date, 'anomaly_label']).size().unstack(fill_value=0)
        
        timeline_data = []
        for date, row in timeline_groups.iterrows():
            # Use .get() safely because sometimes 'Normal' or 'Anomaly' columns might be missing
            timeline_data.append({
                "date": str(date),
                "Normal": int(row.get('Normal', 0)),
                "Anomaly": int(row.get('Anomaly', 0))
            })
            
        timeline_data.sort(key=lambda x: x['date'])

        # --- EXPORT ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum())
        }

        df['timestamp'] = df['timestamp_dt'].astype(str)
        cols_to_keep = ['user_id', 'anomaly_label', 'anomaly_score', 'timestamp', 'country', 
                        'operation', 'file_type', 'file_size', 'subscription_type', 'success']
        
        final_cols = [c for c in cols_to_keep if c in df.columns]
        df_export = df[final_cols].copy()

        pretty_names = {
            'user_id': 'User ID', 'anomaly_label': 'Status', 'anomaly_score': 'Risk Score',
            'timestamp': 'Time', 'country': 'Country', 'operation': 'Action',
            'file_type': 'File Type', 'file_size': 'Size', 'subscription_type': 'Plan',
            'success': 'Success'
        }
        df_export.rename(columns=pretty_names, inplace=True)

        country_map = {0: 'DE', 1: 'FR', 2: 'GB', 3: 'PL', 4: 'UA', 5: 'US'}
        if 'Country' in df_export.columns:
            df_export['Country'] = df_export['Country'].map(country_map).fillna(df_export['Country'])

        del df
        del X_scaled
        gc.collect()

        return jsonify({
            "summary": summary,
            "benchmarks": benchmarks_data,
            "timeline": timeline_data,
            "results": df_export.to_dict(orient="records")
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)