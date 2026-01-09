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
# 🚀 LOAD THE "BRAIN" (MODEL) ONCE
# ==========================================
print("⏳ Loading pre-trained model...")
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("✅ Model loaded successfully! (Fast Mode)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, scaler = None, None

# ==========================================
# 🧠 HELPER: EXPLAINABILITY & CLEANING
# ==========================================
def find_reasons(row, df_stats):
    reasons = []
    if row['file_size'] > (df_stats['avg_size'] * 5): reasons.append("Unusual File Size")
    if 3 <= row['hour'] <= 5: reasons.append("Odd Hours (Night)")
    if row['account_age_days'] < 1: reasons.append("New Account")
    if row['success'] == 0: reasons.append("Failed Operation")
    return ", ".join(reasons) if reasons else "Pattern Deviation"

def preprocess_data(df):
    """ Cleans and prepares the CSV in-place to save memory. """
    # 1. SMART RENAME
    rename_map = {
        'timestamp': ['date', 'time', 'created_at', 'datetime'],
        'user_id': ['id', 'client_id', 'customer_id', 'username'],
        'file_size': ['size', 'bytes', 'length'],
        'operation': ['action', 'activity', 'type'],
        'success': ['status', 'result'],
        'country': ['region', 'geo'],
        'file_type': ['file_format', 'extension'],
        'subscription_type': ['plan', 'tier']
    }
    
    df.columns = [c.lower().strip() for c in df.columns]
    curr_cols = set(df.columns)
    
    for std, alts in rename_map.items():
        if std not in curr_cols:
            match = next((c for alt in alts for c in curr_cols if c == alt), None)
            if match: df.rename(columns={match: std}, inplace=True)

    # 2. FEATURE ENGINEERING
    if 'timestamp' not in df.columns: df['timestamp'] = '2024-01-01'
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    
    # Account Age Logic
    if 'signup_date' in df.columns:
        signup = pd.to_datetime(df['signup_date'], errors='coerce').fillna(df['timestamp_dt'])
        df['account_age_days'] = (df['timestamp_dt'] - signup).dt.days.fillna(0).astype('int16')
    else:
        df['account_age_days'] = np.random.randint(0, 365, size=len(df))

    df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

    # Normalize Columns
    if 'success' in df.columns:
        df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes', 'success']).astype('int8')
    else:
        df['success'] = 1
    
    # --- FIX: HASH USER ID (TEXT -> NUMBER) ---
    # The AI model crashes on text like "User_Admin". We convert it to a number.
    if 'user_id' in df.columns:
        df['user_id_num'] = df['user_id'].astype(str).apply(lambda x: abs(hash(x)) % 100000)
    else:
        df['user_id_num'] = 0

    # Fill Numeric NaNs
    for c in ['file_size', 'storage_limit', 'country', 'operation', 'file_type']:
        if c not in df.columns: df[c] = 0
        else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Note: We use 'user_id_num' instead of 'user_id' for the AI
    features = ['user_id_num', 'operation', 'file_type', 'file_size', 'success', 'subscription_type', 
                'storage_limit', 'country', 'hour', 'account_age_days', 'day_of_week', 'is_weekend']
    return df, features

# ==========================================
# 🔍 MAIN ROUTE: /DETECT
# ==========================================
@app.route('/detect', methods=['POST'])
def detect():
    if not model or not scaler:
        return jsonify({"error": "Server Error: Model not loaded."}), 500

    try:
        if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        sensitivity = float(request.form.get('sensitivity', 50))
        
        try:
            df_original = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
        except:
            return jsonify({"error": "Invalid CSV file."}), 400

        # --- PREPROCESS ---
        try:
            df, feature_cols = preprocess_data(df_original)
        except Exception as e:
            return jsonify({"error": f"Processing Error: {str(e)}"}), 400

        if df.empty: return jsonify({"error": "Empty dataset produced."}), 400

        # --- PREDICT ---
        # Ensure we only pass numeric columns to the AI
        X_scaled = np.nan_to_num(scaler.transform(df[feature_cols]).astype(np.float32))
        raw_scores = model.decision_function(X_scaled)
        
        # Dynamic Thresholding
        contamination = 0.001 + (sensitivity / 100) * 0.1 
        threshold = np.percentile(raw_scores, 100 * contamination)
        critical_threshold = np.percentile(raw_scores, 100 * (contamination * 0.2))
        
        predictions = np.where(raw_scores < threshold, -1, 1)
        custom_anomalies = np.where(predictions == -1, 'Anomaly', 'Normal')

        # Vectorized Risk Level Calculation
        risk_levels = np.select(
            [predictions == 1, raw_scores < critical_threshold, raw_scores < threshold],
            [0, 3, 2], default=1
        )

        # Generate Reasons (Only for Anomalies)
        stats = {'avg_size': df['file_size'].mean()}
        df['Analysis'] = [find_reasons(row, stats) if label == 'Anomaly' else "Normal Activity" 
                          for _, (row, label) in enumerate(zip(df.iloc, custom_anomalies))]

        df['anomaly_score'] = risk_levels
        df['anomaly_label'] = custom_anomalies

        # --- BENCHMARKS (CRASH-PROOF VERSION) ---
        benchmarks_data = []
        try:
            subset_idx = np.random.choice(len(df), min(len(df), 15000), replace=False)
            y_truth = np.where(np.array(custom_anomalies)[subset_idx] == 'Anomaly', 1, 0)
            X_bench = X_scaled[subset_idx]
            
            unique, counts = np.unique(y_truth, return_counts=True)
            min_count = counts.min() if len(counts) > 1 else 0

            # Only run if we have BOTH classes (Normal & Anomaly)
            if len(unique) > 1:
                # Fallback to non-stratified split if data is too scarce
                stratify_strategy = y_truth if min_count > 1 else None
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_bench, y_truth, test_size=0.3, random_state=42, stratify=stratify_strategy
                )

                # 1. Random Forest (Fast Check)
                rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                p, r, f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='binary', zero_division=0)
                benchmarks_data.append({"name": "Random Forest", "Precision": round(p,2), "Recall": round(r,2), "F1": round(f1,2)})

                # 2. Logistic Regression
                lr = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced')
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                p, r, f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='binary', zero_division=0)
                benchmarks_data.append({"name": "Logistic Regression", "Precision": round(p,2), "Recall": round(r,2), "F1": round(f1,2)})

        except Exception as e:
            print(f"Benchmark skipped: {e}")

        # --- EXPORT ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum()),
            "sensitivity_used": sensitivity
        }

        # Recover Readable Strings
        df['timestamp'] = df['timestamp_dt'].astype(str)
        mapping_data = {
            'country': {0: 'Other', 1: 'FR', 2: 'GB', 3: 'PL', 4: 'UA', 5: 'US'},
            'operation': {0: 'delete', 1: 'download', 2: 'modify', 3: 'upload'},
            'file_type': {0: 'archive', 1: 'document', 2: 'photo', 3: 'video'},
            'subscription_type': {0: 'business', 1: 'free', 2: 'premium'},
            'success': {1: 'Success', 0: 'Failed'}
        }
        for col, mp in mapping_data.items():
            df[col] = df[col].map(mp).fillna(df[col])

        # Timeline
        timeline_groups = df.groupby([df['timestamp_dt'].dt.date, 'anomaly_label']).size().unstack(fill_value=0)
        timeline_data = [{"date": str(d), "Normal": int(r.get('Normal', 0)), "Anomaly": int(r.get('Anomaly', 0))} 
                         for d, r in timeline_groups.iterrows()]
        timeline_data.sort(key=lambda x: x['date'])

        # Final Clean
        df_export = df.drop(columns=['timestamp_dt', 'user_id_num'], errors='ignore')
        df_export.rename(columns={
            'user_id': 'User ID', 'anomaly_label': 'Status', 'anomaly_score': 'Risk Score',
            'timestamp': 'Time', 'operation': 'Action', 'file_size': 'Size', 
            'account_age_days': 'Account Age', 'Analysis': 'AI Reasoning'
        }, inplace=True)

        # Truncate Response
        df_final = pd.concat([
            df_export[df_export['Status'] == 'Anomaly'],
            df_export[df_export['Status'] == 'Normal'].head(5000 - len(df_export[df_export['Status'] == 'Anomaly']))
        ]).fillna("")

        del df, X_scaled, df_original
        gc.collect()

        return jsonify({
            "summary": summary,
            "benchmarks": benchmarks_data,
            "timeline": timeline_data,
            "results": df_final.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

# ==========================================
# 📥 ROUTE: DOWNLOAD SMART TEMPLATE
# ==========================================
@app.route('/template', methods=['GET'])
def download_template():
    data = {
        'timestamp': ['2024-01-01 09:00:00', '2024-01-01 10:00:00', '2024-01-01 03:00:00', '2024-01-01 03:15:00'],
        'user_id': ['User123', 'User123', 'HackerX', 'HackerX'],
        'file_size': [1024, 2048, 999999999, 888888888],
        'operation': ['upload', 'download', 'delete', 'modify'],
        'success': ['True', 'True', 'False', 'False'],
        'country': ['US', 'US', 'Unknown', 'Unknown'],
        'file_type': ['document', 'photo', 'exe', 'sh'],
        'subscription_type': ['premium', 'premium', 'free', 'free']
    }
    output = BytesIO()
    pd.DataFrame(data).to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='CloudRadar_Template.csv')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)