from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import gc
import time 
from io import BytesIO

app = Flask(__name__)
CORS(app)

app.json.sort_keys = False 

# ==========================================
# 🚀 LOAD MODEL
# ==========================================
print("⏳ Loading pre-trained model...")
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# ==========================================
# 🧠 HELPER: EXPLAINABILITY
# ==========================================
def find_reasons(row, df_stats):
    reasons = []
    # Safe checks with .get to avoid crashes
    val_size = row.get('file_size', 0)
    val_hour = row.get('hour', 0)
    val_age = row.get('account_age_days', 0)
    val_success = row.get('success', 1)

    if val_size > 0 and val_size > (df_stats['avg_size'] * 5):
        reasons.append("Unusual File Size")
    if 3 <= val_hour <= 5:
        reasons.append("Odd Hours (Night)")
    if val_age < 1:
        reasons.append("New Account")
    if val_success == 0:
        reasons.append("Failed Operation")
    
    if not reasons: reasons.append("Pattern Deviation")
    return ", ".join(reasons)

# ==========================================
# 🛠️ HELPER: PREPROCESSING (With Fuzzy Match)
# ==========================================
def preprocess_data(df):
    # 1. FUZZY COLUMN MATCHING
    rename_map = {
        'timestamp': ['date', 'time', 'created', 'moment', 'clock'],
        'user_id': ['user', 'client', 'id', 'account', 'customer', 'login'],
        'file_size': ['size', 'byte', 'storage', 'len', 'weight'],
        'operation': ['action', 'activ', 'event', 'type', 'method', 'op'],
        'success': ['status', 'success', 'result', 'pass', 'code'],
        'country': ['countr', 'geo', 'region', 'loc', 'place'],
        'file_type': ['format', 'ext', 'type', 'mime'],
        'subscription_type': ['plan', 'tier', 'level', 'sub']
    }

    # Normalize input columns
    df.columns = [str(c).lower().strip().replace('_', '').replace(' ', '') for c in df.columns]
    curr_cols = list(df.columns)
    cols_to_rename = {}

    for standard_name, keywords in rename_map.items():
        if standard_name in curr_cols: continue
        
        # Find best match (first column that contains a keyword)
        match = next((col for col in curr_cols if any(k in col for k in keywords)), None)
        if match:
            cols_to_rename[match] = standard_name
            curr_cols.remove(match) 

    if cols_to_rename:
        df.rename(columns=cols_to_rename, inplace=True)

    # 2. FEATURE ENGINEERING
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    else:
        df['timestamp_dt'] = pd.Timestamp('2024-01-01')

    # Random Account Age (Simulation)
    df['account_age_days'] = np.random.randint(0, 365, size=len(df)).astype('int16')

    # Time Features
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')

    # Success Flag
    if 'success' in df.columns:
        df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes', 'success']).astype('int8')
    else:
        df['success'] = 1

    # Fill Numeric Gaps
    req_cols = ['file_size', 'storage_limit', 'subscription_type', 'country', 'operation', 'user_id', 'file_type', 'hour']
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

# ==========================================
# 🔍 MAIN ROUTE: /DETECT
# ==========================================
@app.route('/detect', methods=['POST'])
def detect():
    if not model: return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    
    try:
        # 1. LOAD CSV
        df_original = pd.read_csv(request.files['file'], encoding='utf-8-sig', low_memory=False)
        
        # 2. PREPROCESS
        try:
            df, feature_cols = preprocess_data(df_original)
        except Exception as e:
            return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400

        # 3. PREDICT
        X_scaled = scaler.transform(df[feature_cols]).astype(np.float32)
        X_scaled = np.nan_to_num(X_scaled)
        
        # Sensitivity Logic
        sens = float(request.form.get('sensitivity', 50))
        contamination = 0.001 + (sens / 100) * 0.1
        
        raw_scores = model.decision_function(X_scaled)
        threshold = np.percentile(raw_scores, 100 * contamination)
        predictions = np.where(raw_scores < threshold, -1, 1) # -1 is Anomaly

        # 4. ORGANIZE RESULTS
        df['anomaly_label'] = np.where(predictions == -1, 'Anomaly', 'Normal')
        
        # Calculate Risk Levels
        crit_thresh = np.percentile(raw_scores, 100 * (contamination * 0.2))
        df['anomaly_score'] = [3 if s < crit_thresh else 2 if s < threshold else 0 for s in raw_scores]

        # Generate Reasoning
        stats = {'avg_size': df['file_size'].mean()}
        analysis_col = []
        records = df.to_dict('records')
        for i, row in enumerate(records):
            if row['anomaly_label'] == 'Anomaly':
                analysis_col.append(find_reasons(row, stats))
            else:
                analysis_col.append("Normal Activity")
        df['Analysis'] = analysis_col

        # 5. BENCHMARKS (FIXED CRASH HERE)
        benchmarks_data = []
        try:
            subset_size = min(len(df), 15000)
            idx = np.random.choice(len(df), subset_size, replace=False)
            y_truth = np.where(df['anomaly_label'].iloc[idx] == 'Anomaly', 1, 0)
            X_bench = X_scaled[idx]
            
            # --- THE SAFETY CHECK ---
            # We only run benchmarks if there are at least 2 anomalies and 2 normal rows
            unique, counts = np.unique(y_truth, return_counts=True)
            if len(unique) > 1 and all(c >= 2 for c in counts):
                X_tr, X_te, y_tr, y_te = train_test_split(X_bench, y_truth, test_size=0.3, stratify=y_truth)
                
                for name, clf in [("Random Forest", RandomForestClassifier(n_estimators=50)), 
                                  ("Logistic Regression", LogisticRegression(solver='liblinear'))]:
                    clf.fit(X_tr, y_tr)
                    p, r, f, _ = precision_recall_fscore_support(y_te, clf.predict(X_te), average='binary', zero_division=0)
                    benchmarks_data.append({"name": name, "Precision": round(p,2), "Recall": round(r,2), "F1": round(f,2)})
            else:
                # Skip silently or add a note
                benchmarks_data.append({
                    "name": "Dataset too small for benchmark (Needs >2 anomalies)", 
                    "Precision": 0, "Recall": 0, "F1": 0
                })
        except Exception as e:
            print(f"Benchmark Warning: {e}") 

        # 6. EXPORT PREP
        df['timestamp'] = df['timestamp_dt'].astype(str)
        
        # Timeline
        timeline = df.groupby([df['timestamp_dt'].dt.date.astype(str), 'anomaly_label']).size().unstack(fill_value=0).reset_index()
        timeline.rename(columns={'timestamp_dt': 'date'}, inplace=True)

        # Cleanup columns for frontend
        out_map = {
            'user_id': 'User ID', 'anomaly_label': 'Status', 'anomaly_score': 'Risk Score',
            'timestamp': 'Time', 'country': 'Country', 'operation': 'Action',
            'file_size': 'Size', 'Analysis': 'AI Reasoning'
        }
        df.rename(columns=out_map, inplace=True)
        
        keep_cols = list(out_map.values())
        final_df = df[keep_cols]

        # Sort: Anomalies first
        df_anom = final_df[final_df['Status'] == 'Anomaly']
        df_norm = final_df[final_df['Status'] == 'Normal'].head(5000)
        final_data = pd.concat([df_anom, df_norm])

        del df, X_scaled, raw_scores
        gc.collect()

        return jsonify({
            "summary": {"total_rows": len(df_original), "anomalies": len(df_anom)},
            "benchmarks": benchmarks_data,
            "timeline": timeline.to_dict(orient='records'),
            "results": final_data.fillna("").to_dict(orient="records")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

# 📥 TEMPLATE DOWNLOAD (NOW 50 ROWS)
@app.route('/template', methods=['GET'])
def download_template():
    # We create 50 rows to ensure benchmarks always run safely
    # 45 Normal, 5 Anomalies
    data = {
        'Timestamp': ['2024-01-01 09:00']*45 + ['2024-01-01 03:00', '2024-01-01 03:15', '2024-01-01 12:00', '2024-01-01 12:05', '2024-01-01 23:59'],
        'User_Identity': ['User']*45 + ['HACKER_X', 'HACKER_X', 'Inside_Job', 'Inside_Job', 'Unknown'],
        'File_Size_Bytes': [1024]*45 + [100, 100, 999999999, 999999999, 500],
        'Activity_Type': ['upload']*45 + ['download', 'delete', 'download', 'download', 'login'],
        'Is_Success': ['True']*45 + ['True', 'False', 'True', 'True', 'False']
    }
    out = BytesIO()
    pd.DataFrame(data).to_csv(out, index=False)
    out.seek(0)
    return send_file(out, mimetype='text/csv', as_attachment=True, download_name='Smart_Audit_Template_Large.csv')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)