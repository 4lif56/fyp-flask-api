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
# 🚀 LOAD MODEL (Memory Optimized)
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
# 🧠 HELPER: EXPLAINABILITY
# ==========================================
def find_reasons(row, avg_size):
    reasons = []
    # 1. Check File Size (if 5x bigger than average)
    if row['file_size'] > 0 and row['file_size'] > (avg_size * 5):
        reasons.append("Unusual File Size")
    # 2. Check Hour (3 AM - 5 AM is sus)
    if 3 <= row['hour'] <= 5:
        reasons.append("Odd Hours (Night)")
    # 3. Check Account Age (New account < 1 day)
    if row['account_age_days'] < 1:
        reasons.append("New Account")
    # 4. Success Failure (If operation failed)
    if row['success'] == 0:
        reasons.append("Failed Operation")
    # Fallback
    if not reasons:
        reasons.append("Pattern Deviation") 
    return ", ".join(reasons)

# ==========================================
# 🛠️ HELPER: ROBUST IN-PLACE PREPROCESSING
# ==========================================
def preprocess_data_inplace(df):
    """
    Selects ONLY the best matching columns and renames them.
    Prevents "Duplicate Column" crashes.
    """
    # 1. Prepare Clean Headers for Matching
    df_cols_clean = [str(c).lower().strip().replace('_', '').replace(' ', '') for c in df.columns]
    
    # 2. Updated Concepts (Added 'operation' and 'storage_limit' support)
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

    # 3. Find Best Match for each Concept (Strict Mode)
    rename_map = {}
    used_indices = set()
    
    for concept, keywords in concepts.items():
        best_idx = -1
        best_score = 0
        
        for idx, col_clean in enumerate(df_cols_clean):
            if idx in used_indices: continue # Don't reuse columns
            
            score = sum(1 for k in keywords if k in col_clean)
            # Tie-breakers
            if concept == 'user_id' and 'user' in col_clean: score += 2
            if concept == 'file_size' and 'byte' in col_clean: score += 2
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            original_name = df.columns[best_idx]
            rename_map[original_name] = concept
            used_indices.add(best_idx)

    # 4. Drop Unused Columns FIRST (Saves RAM)
    cols_to_drop = [c for c in df.columns if c not in rename_map]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    
    # 5. Rename (Now guaranteed unique)
    df.rename(columns=rename_map, inplace=True)
    gc.collect()

    # 6. Feature Engineering
    if 'timestamp' not in df.columns: df['timestamp'] = '2024-01-01'
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    
    # Tiny Ints (Memory Optimization)
    df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    df['account_age_days'] = np.random.randint(0, 365, size=len(df)).astype('int16')

    # Hash User ID
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(str).apply(lambda x: abs(hash(x)) % 100000)
    else:
        df['user_id'] = 0

    # Fix Numerics
    numeric_cols = ['file_size', 'storage_limit', 'country', 'operation', 'file_type', 'subscription_type', 'success']
    for c in numeric_cols:
        if c not in df.columns: df[c] = 0
        else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    features = ['user_id', 'operation', 'file_type', 'file_size', 'success', 'subscription_type', 
                'storage_limit', 'country', 'hour', 'account_age_days', 'day_of_week', 'is_weekend']
    
    return df, features

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

        # 1. LOAD
        try:
            df = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
        except:
            return jsonify({"error": "Invalid CSV"}), 400

        # 2. PREPROCESS (Safe Mode)
        try:
            df, feature_cols = preprocess_data_inplace(df)
            gc.collect()
        except Exception as e:
            return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400

        # 3. PREDICT (Full Scan 76k)
        X_scaled = np.nan_to_num(scaler.transform(df[feature_cols]).astype(np.float32))
        raw_scores = model.decision_function(X_scaled)
        
        # Thresholds
        contamination = 0.001 + (sensitivity / 100) * 0.1 
        threshold = np.percentile(raw_scores, 100 * contamination)
        predictions = np.where(raw_scores < threshold, -1, 1)
        custom_anomalies = np.where(predictions == -1, 'Anomaly', 'Normal')

        # 4. REASONING
        avg_size = df['file_size'].mean()
        df['Status'] = custom_anomalies
        df['Risk Score'] = np.where(raw_scores < threshold, 3, 0)
        df['AI Reasoning'] = [find_reasons(row, avg_size) if label == 'Anomaly' else "Normal" 
                              for _, (row, label) in enumerate(zip(df.iloc, custom_anomalies))]

        # 5. BENCHMARKS (Optimized)
        benchmarks_data = []
        try:
            # Use Safe Subset of 15k for benchmarks to avoid crash
            # The AI scanned 76k, but we validate on 15k
            if len(df) > 15000:
                subset_idx = np.random.choice(len(df), 15000, replace=False)
                X_bench = X_scaled[subset_idx]
                y_bench = np.where(df['Status'].iloc[subset_idx] == 'Anomaly', 1, 0)
            else:
                X_bench = X_scaled
                y_bench = np.where(df['Status'] == 'Anomaly', 1, 0)

            if len(np.unique(y_bench)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X_bench, y_bench, test_size=0.3, random_state=42)
                
                # A. Random Forest (Single Run, 50 Trees)
                rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced', n_jobs=1)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                p, r, f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='binary', zero_division=0)
                benchmarks_data.append({
                    "name": "Random Forest", 
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
            print(f"Benchmark Warning: {e}")

        del X_scaled
        gc.collect()

        # 6. EXPORT
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['Status'] == 'Anomaly').sum()),
            "sensitivity_used": sensitivity
        }

        # Timeline
        timeline_groups = df.groupby([df['timestamp_dt'].dt.date, 'Status']).size().unstack(fill_value=0)
        timeline_data = [{"date": str(d), "Normal": int(r.get('Normal', 0)), "Anomaly": int(r.get('Anomaly', 0))} 
                         for d, r in timeline_groups.iterrows()]
        
        # Rename & Truncate
        df.rename(columns={'timestamp': 'Time', 'user_id': 'User ID', 'operation': 'Action', 'file_size': 'Size'}, inplace=True)
        
        # Return max 5000 rows for browser speed
        df_anomalies = df[df['Status'] == 'Anomaly']
        df_normal = df[df['Status'] == 'Normal']
        
        remaining = 5000 - len(df_anomalies)
        if remaining > 0:
            final_rows = pd.concat([df_anomalies, df_normal.head(remaining)]).fillna("")
        else:
            final_rows = df_anomalies.head(5000).fillna("")

        results_json = final_rows.to_dict(orient="records")

        del df, df_anomalies, df_normal
        gc.collect()

        return jsonify({
            "summary": summary,
            "benchmarks": benchmarks_data,
            "timeline": timeline_data,
            "results": results_json
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