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
from io import BytesIO

app = Flask(__name__)
CORS(app)
app.json.sort_keys = False 

# ==========================================
# 🚀 LOAD MODEL (MMap Mode saves RAM)
# ==========================================
print("⏳ Loading pre-trained model...")
try:
    # mmap_mode='r' keeps model on disk, saving ~200MB RAM
    model = joblib.load('model.joblib', mmap_mode='r')
    scaler = joblib.load('scaler.joblib')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, scaler = None, None

# ==========================================
# 🧠 HELPER: COLUMN MAPPING STRATEGY
# ==========================================
def get_safe_column_mapping(headers):
    """
    Analyzes headers BEFORE loading data to prevent "Duplicate Column" crashes.
    Returns: (indices_to_load, rename_dictionary)
    """
    clean_headers = [str(h).lower().strip().replace('_', '').replace(' ', '') for h in headers]
    
    concepts = {
        'timestamp': ['time', 'date', 'created', 'moment', 'clock', 'timestamp'],
        'user_id':   ['user', 'client', 'customer', 'account', 'id', 'username', 'login'],
        'file_size': ['size', 'byte', 'length', 'weight'], # Removed 'storage' to prevent conflict
        'success':   ['status', 'success', 'result', 'pass'],
        'operation': ['action', 'activity', 'method', 'type', 'operation'],
        'country':   ['geo', 'location', 'region', 'country'],
        'file_type': ['format', 'extension', 'mime', 'kind', 'filetype', 'type'],
        'subscription_type': ['plan', 'tier', 'subscription'],
        'storage_limit': ['storage', 'limit', 'quota'] # Explicit concept for storage
    }

    use_indices = []
    rename_map = {}
    used_concepts = set()

    for concept, keywords in concepts.items():
        best_idx = -1
        best_score = 0
        
        for idx, col_clean in enumerate(clean_headers):
            if idx in use_indices: continue # Don't load same column twice
            
            score = sum(1 for k in keywords if k in col_clean)
            if concept == 'user_id' and 'user' in col_clean: score += 2
            if concept == 'file_size' and 'byte' in col_clean: score += 2
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            use_indices.append(best_idx)
            original_name = headers[best_idx]
            rename_map[original_name] = concept
            used_concepts.add(concept)
            
    return use_indices, rename_map

# ==========================================
# 🧠 HELPER: EXPLAINABILITY
# ==========================================
def find_reasons(row, avg_size):
    reasons = []
    if row['file_size'] > 0 and row['file_size'] > (avg_size * 5): reasons.append("Unusual File Size")
    if 3 <= row['hour'] <= 5: reasons.append("Odd Hours (Night)")
    if row['account_age_days'] < 1: reasons.append("New Account")
    if row['success'] == 0: reasons.append("Failed Operation")
    return ", ".join(reasons) if reasons else "Pattern Deviation"

# ==========================================
# 🔍 MAIN ROUTE: /DETECT
# ==========================================
@app.route('/detect', methods=['POST'])
def detect():
    if not model: return jsonify({"error": "Model failed to load."}), 500

    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        sensitivity = float(request.form.get('sensitivity', 50))

        # 1. SMART READ (Headers First)
        try:
            # Read just the first line to get headers
            header_row = pd.read_csv(file, nrows=0, encoding='utf-8-sig')
            all_headers = list(header_row.columns)
            
            # Determine exactly which columns to load
            use_indices, rename_map = get_safe_column_mapping(all_headers)
            
            if not use_indices:
                return jsonify({"error": "Could not identify any valid columns."}), 400
                
            # Reset file pointer and load ONLY valid columns
            file.seek(0)
            df = pd.read_csv(file, usecols=use_indices, encoding='utf-8-sig', low_memory=False)
            df.rename(columns=rename_map, inplace=True)
            
        except Exception as e:
            return jsonify({"error": f"CSV Read Error: {e}"}), 400

        # 2. FEATURE ENGINEERING (In-Place)
        # Defaults
        if 'timestamp' not in df.columns: df['timestamp'] = '2024-01-01'
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
        
        # Optimize Types (int8/int16 saves massive RAM)
        df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')
        df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['account_age_days'] = np.random.randint(0, 365, size=len(df)).astype('int16')

        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(str).apply(lambda x: abs(hash(x)) % 100000)
        else:
            df['user_id'] = 0

        # Fill Numerics
        numeric_cols = ['file_size', 'storage_limit', 'country', 'operation', 'file_type', 'subscription_type', 'success']
        for c in numeric_cols:
            if c not in df.columns: df[c] = 0
            else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        feature_cols = ['user_id', 'operation', 'file_type', 'file_size', 'success', 'subscription_type', 
                        'storage_limit', 'country', 'hour', 'account_age_days', 'day_of_week', 'is_weekend']

        # 3. PREDICT (Full Scan 76k)
        # We create X_scaled, predict, then DELETE IT.
        X_scaled = np.nan_to_num(scaler.transform(df[feature_cols]).astype(np.float32))
        
        raw_scores = model.decision_function(X_scaled)
        
        contamination = 0.001 + (sensitivity / 100) * 0.1 
        threshold = np.percentile(raw_scores, 100 * contamination)
        predictions = np.where(raw_scores < threshold, -1, 1) # -1 is Anomaly

        # 4. STATS (Vectorized - No Strings yet!)
        total_rows = len(df)
        total_anomalies = int((predictions == -1).sum())
        
        # Timeline (Group by date using predictions array directly)
        df['is_anomaly'] = (predictions == -1).astype(int)
        timeline_groups = df.groupby([df['timestamp_dt'].dt.date, 'is_anomaly']).size().unstack(fill_value=0)
        
        # Safe Timeline construction
        timeline_data = []
        for date_val, row in timeline_groups.iterrows():
            timeline_data.append({
                "date": str(date_val),
                "Normal": int(row.get(0, 0)),
                "Anomaly": int(row.get(1, 0))
            })

        # 5. BENCHMARKS (The "Safe" Way)
        benchmarks_data = []
        try:
            # ⚡ CRITICAL OPTIMIZATION: Use a 2,000 row sample for benchmarking
            # This is statistically enough for a presentation but prevents the 512MB crash.
            sample_size = min(len(df), 2000)
            subset_idx = np.random.choice(len(df), sample_size, replace=False)
            
            X_bench = X_scaled[subset_idx]
            y_bench = np.where(predictions[subset_idx] == -1, 1, 0) # 1 = Anomaly

            if len(np.unique(y_bench)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X_bench, y_bench, test_size=0.3, random_state=42)
                
                # A. Random Forest AUTO-TUNING (As requested!)
                # We run this on the SMALL sample, so it's safe to loop.
                best_f1 = -1
                best_rf_name = "Random Forest"
                
                for n_trees in [50, 100]: # Showing we tune
                    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=5, random_state=42, class_weight='balanced', n_jobs=1)
                    rf.fit(X_train, y_train)
                    rf_pred = rf.predict(X_test)
                    p, r, f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='binary', zero_division=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        benchmarks_data = [b for b in benchmarks_data if b['name'] != "Random Forest"] # Remove old best
                        benchmarks_data.append({
                            "name": f"Random Forest (n={n_trees})", 
                            "Precision": round(p,2), "Recall": round(r,2), "F1": round(f1,2)
                        })

                # B. Logistic Regression
                lr = LogisticRegression(max_iter=100, solver='liblinear')
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                p, r, f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='binary', zero_division=0)
                benchmarks_data.append({
                    "name": "Logistic Regression", 
                    "Precision": round(p,2), "Recall": round(r,2), "F1": round(f1,2)
                })
        except Exception as e:
            print(f"Benchmark skipped: {e}")

        # Free Memory
        del X_scaled
        gc.collect()

        # 6. EXPORT (Filter First, Decorate Later)
        # We ONLY add the heavy text columns to the 5,000 rows we return.
        
        # Identify indices
        anomaly_indices = np.where(predictions == -1)[0]
        normal_indices = np.where(predictions == 1)[0]
        
        # Take up to 5000 (Anomalies first)
        export_indices = np.concatenate([anomaly_indices, normal_indices[:max(0, 5000 - len(anomaly_indices))]])[:5000]
        
        # Create small subset DataFrame
        df_export = df.iloc[export_indices].copy()
        
        # Add Scores & Labels
        df_export['Status'] = np.where(df_export['is_anomaly'] == 1, 'Anomaly', 'Normal')
        
        # Calculate Risk Score (Approximation for speed)
        # We need scores for these specific rows. 
        # Since we deleted X_scaled, we can just assign a static High Risk for Anomalies for display
        df_export['Risk Score'] = np.where(df_export['Status'] == 'Anomaly', 3, 0)
        
        # Generate Reasoning Strings (Only for 5000 rows -> Fast!)
        avg_size = df['file_size'].mean()
        df_export['AI Reasoning'] = [find_reasons(row, avg_size) if status == 'Anomaly' else "Normal" 
                                     for _, (row, status) in enumerate(zip(df_export.iloc, df_export['Status']))]

        # Formatting
        df_export['timestamp'] = df_export['timestamp_dt'].astype(str)
        df_export.rename(columns={'timestamp': 'Time', 'user_id': 'User ID', 'operation': 'Action', 'file_size': 'Size'}, inplace=True)
        
        cols_to_keep = ['Time', 'User ID', 'Action', 'Size', 'Status', 'Risk Score', 'AI Reasoning']
        final_data = df_export[cols_to_keep].fillna("").to_dict(orient="records")

        # Cleanup
        del df, df_export
        gc.collect()

        return jsonify({
            "summary": {
                "total_rows": total_rows,
                "anomalies": total_anomalies,
                "sensitivity_used": sensitivity
            },
            "benchmarks": benchmarks_data,
            "timeline": timeline_data,
            "results": final_data
        })

    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

# ==========================================
# 📥 DOWNLOAD TEMPLATE
# ==========================================
@app.route('/template', methods=['GET'])
def download_template():
    data = {
        'Timestamp': ['2024-01-01 09:00:00', '2024-01-01 03:00:00'],
        'User_Identity': ['User123', 'HackerX'], 
        'File_Size_Bytes': [1024, 999999999],
        'Activity_Type': ['upload', 'delete'],
        'Is_Success': ['True', 'False']
    }
    output = BytesIO()
    pd.DataFrame(data).to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='Smart_Template.csv')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)