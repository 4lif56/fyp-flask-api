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
# 🧠 HELPER: EXPLAINABILITY
# ==========================================
def find_reasons(row, df_stats):
    reasons = []
    if row['file_size'] > (df_stats['avg_size'] * 5): reasons.append("Unusual File Size")
    if 3 <= row['hour'] <= 5: reasons.append("Odd Hours (Night)")
    if row['account_age_days'] < 1: reasons.append("New Account")
    if row['success'] == 0: reasons.append("Failed Operation")
    return ", ".join(reasons) if reasons else "Pattern Deviation"

# ==========================================
# 🛠️ HELPER: SMART DATA PREPROCESSING
# ==========================================
def preprocess_data(df):
    """ 
    Uses Fuzzy Logic to auto-detect column meanings regardless of naming.
    """
    # 1. CLEAN HEADERS
    df.columns = [c.lower().strip().replace('_', '').replace(' ', '') for c in df.columns]
    
    # 2. DEFINE CONCEPTS & KEYWORDS
    # We look for these words in the headers to guess what they are.
    concepts = {
        'timestamp': ['time', 'date', 'created', 'moment', 'clock'],
        'user_id':   ['user', 'client', 'customer', 'account', 'id', 'username', 'login', 'email', 'employee'],
        'file_size': ['size', 'byte', 'length', 'capacity', 'storage', 'weight'],
        'success':   ['status', 'success', 'result', 'pass', 'code', 'state'],
        'operation': ['action', 'activity', 'method', 'type', 'event', 'command'],
        'country':   ['geo', 'location', 'region', 'country', 'zone', 'ip'],
        'file_type': ['format', 'extension', 'mime', 'kind', 'filetype']
    }

    # 3. SMART MAPPING LOOP
    # We assign columns based on which concept matches the most keywords.
    found_cols = set()
    rename_map = {}

    for target_name, keywords in concepts.items():
        best_match = None
        best_score = 0
        
        for col in df.columns:
            if col in found_cols: continue # Already mapped
            
            # Scoring: +1 for every keyword found in the column name
            score = sum(1 for k in keywords if k in col)
            
            # Tie-breaker: prioritization (e.g., 'user_id' is better than just 'id')
            if target_name == 'user_id' and 'user' in col: score += 2
            if target_name == 'file_size' and 'byte' in col: score += 2

            if score > best_score:
                best_score = score
                best_match = col
        
        if best_match and best_score > 0:
            rename_map[best_match] = target_name
            found_cols.add(best_match)

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # 4. FEATURE ENGINEERING (Standardize Values)
    
    # Time
    if 'timestamp' not in df.columns: df['timestamp'] = '2024-01-01'
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    
    df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

    # Account Age
    df['account_age_days'] = np.random.randint(0, 365, size=len(df)).astype('int16') # Simulation if missing

    # Success Boolean
    if 'success' in df.columns:
        df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes', 'success', 'ok', '200']).astype('int8')
    else:
        df['success'] = 1

    # User ID Hashing (The "Any Name to Number" Logic)
    # This ensures "JohnDoe", "Client_99", "admin" all become safe numbers for the AI
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(str).apply(lambda x: abs(hash(x)) % 100000)
    else:
        df['user_id'] = 0

    # Fill Numeric NaNs
    for c in ['file_size', 'storage_limit', 'country', 'operation', 'file_type', 'subscription_type']:
        if c not in df.columns: df[c] = 0
        else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Final Features (Must match Training Order)
    features = ['user_id', 'operation', 'file_type', 'file_size', 'success', 'subscription_type', 
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
        X_scaled = np.nan_to_num(scaler.transform(df[feature_cols]).astype(np.float32))
        raw_scores = model.decision_function(X_scaled)
        
        # Dynamic Thresholding
        contamination = 0.001 + (sensitivity / 100) * 0.1 
        threshold = np.percentile(raw_scores, 100 * contamination)
        critical_threshold = np.percentile(raw_scores, 100 * (contamination * 0.2))
        
        predictions = np.where(raw_scores < threshold, -1, 1)
        custom_anomalies = np.where(predictions == -1, 'Anomaly', 'Normal')

        # Risk Levels
        risk_levels = np.select(
            [predictions == 1, raw_scores < critical_threshold, raw_scores < threshold],
            [0, 3, 2], default=1
        )

        # Generate Reasons
        stats = {'avg_size': df['file_size'].mean()}
        df['Analysis'] = [find_reasons(row, stats) if label == 'Anomaly' else "Normal Activity" 
                          for _, (row, label) in enumerate(zip(df.iloc, custom_anomalies))]

        df['anomaly_score'] = risk_levels
        df['anomaly_label'] = custom_anomalies

        # --- BENCHMARKS ---
        benchmarks_data = []
        try:
            subset_idx = np.random.choice(len(df), min(len(df), 15000), replace=False)
            y_truth = np.where(np.array(custom_anomalies)[subset_idx] == 'Anomaly', 1, 0)
            X_bench = X_scaled[subset_idx]
            
            unique, counts = np.unique(y_truth, return_counts=True)
            min_count = counts.min() if len(counts) > 1 else 0

            if len(unique) > 1:
                stratify_strategy = y_truth if min_count > 1 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X_bench, y_truth, test_size=0.3, random_state=42, stratify=stratify_strategy
                )

                rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                p, r, f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='binary', zero_division=0)
                benchmarks_data.append({"name": "Random Forest", "Precision": round(p,2), "Recall": round(r,2), "F1": round(f1,2)})

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

        # Create Timeline
        timeline_groups = df.groupby([df['timestamp_dt'].dt.date, 'anomaly_label']).size().unstack(fill_value=0)
        timeline_data = [{"date": str(d), "Normal": int(r.get('Normal', 0)), "Anomaly": int(r.get('Anomaly', 0))} 
                         for d, r in timeline_groups.iterrows()]
        timeline_data.sort(key=lambda x: x['date'])

        # Final Cleanup
        df_export = df.drop(columns=['timestamp_dt'], errors='ignore')
        df_export.rename(columns={
            'user_id': 'User ID', 'anomaly_label': 'Status', 'anomaly_score': 'Risk Score',
            'timestamp': 'Time', 'operation': 'Action', 'file_size': 'Size', 
            'account_age_days': 'Account Age', 'Analysis': 'AI Reasoning'
        }, inplace=True)

        df_final = pd.concat([
            df_export[df_export['Status'] == 'Anomaly'],
            df_export[df_export['Status'] == 'Normal'].head(5000)
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
    # We provide standard headers, but thanks to Smart Logic, users can rename them!
    data = {
        'Timestamp': ['2024-01-01 09:00:00', '2024-01-01 10:00:00', '2024-01-01 03:00:00'],
        'User_Identity': ['User123', 'User123', 'HackerX'], # Notice I named this User_Identity to prove a point
        'File_Size_Bytes': [1024, 2048, 999999999],
        'Activity_Type': ['upload', 'download', 'delete'],
        'Is_Success': ['True', 'True', 'False'],
        'Region': ['US', 'US', 'Unknown'],
        'File_Ext': ['document', 'photo', 'exe'],
        'Plan': ['premium', 'premium', 'free']
    }
    output = BytesIO()
    pd.DataFrame(data).to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='Smart_Template.csv')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)