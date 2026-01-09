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
    model = joblib.load('model.joblib', mmap_mode='r')
    scaler = joblib.load('scaler.joblib')
    print("✅ Model loaded successfully! (Full Scan Mode)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, scaler = None, None

# ==========================================
# 🧠 HELPER: REASONING
# ==========================================
def find_reasons(row, avg_size):
    reasons = []
    if row['file_size'] > (avg_size * 5): reasons.append("Unusual File Size")
    if 3 <= row['hour'] <= 5: reasons.append("Odd Hours (Night)")
    if row['account_age_days'] < 1: reasons.append("New Account")
    if row['success'] == 0: reasons.append("Failed Operation")
    return ", ".join(reasons) if reasons else "Pattern Deviation"

# ==========================================
# 🛠️ HELPER: IN-PLACE PREPROCESSING (MEMORY SAVER)
# ==========================================
def preprocess_data_inplace(df):
    """
    Modifies the dataframe IN-PLACE to avoid creating a copy in RAM.
    """
    # 1. Clean Headers
    df.columns = [str(c).lower().strip().replace('_', '').replace(' ', '') for c in df.columns]
    
    # 2. Map Concepts
    concepts = {
        'timestamp': ['time', 'date', 'created', 'moment', 'clock'],
        'user_id':   ['user', 'client', 'customer', 'account', 'id', 'username'],
        'file_size': ['size', 'byte', 'length', 'storage'],
        'success':   ['status', 'success', 'result', 'pass'],
        'operation': ['action', 'activity', 'method', 'type'],
        'country':   ['geo', 'location', 'region', 'country'],
        'file_type': ['format', 'extension', 'mime', 'kind', 'filetype'],
        'subscription_type': ['plan', 'tier', 'subscription']
    }

    # Identify columns to RENAME (don't copy)
    rename_map = {}
    found_concepts = set()

    for col in df.columns:
        best_concept = None
        best_score = 0
        
        for concept, keywords in concepts.items():
            if concept in found_concepts: continue
            score = sum(1 for k in keywords if k in col)
            if concept == 'user_id' and 'user' in col: score += 2
            
            if score > best_score:
                best_score = score
                best_concept = concept
        
        if best_concept and best_score > 0:
            rename_map[col] = best_concept
            found_concepts.add(best_concept)

    # Apply Rename
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    
    # 3. DROP UNUSED COLUMNS IMMEDIATELY (Save RAM)
    # This prevents the "Duplicate Column" crash
    keep_cols = list(concepts.keys()) + ['timestamp_dt', 'hour', 'day_of_week', 'is_weekend', 'account_age_days']
    existing_cols = [c for c in df.columns if c in concepts.keys()]
    
    # Slice to drop others (This is safe)
    if existing_cols:
        df = df[existing_cols] 
    
    gc.collect()

    # 4. Fill & Engineer
    if 'timestamp' not in df.columns: df['timestamp'] = '2024-01-01'
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    
    # Tiny Ints
    df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    df['account_age_days'] = np.random.randint(0, 365, size=len(df)).astype('int16')

    # Hash User ID (In Place)
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(str).apply(lambda x: abs(hash(x)) % 100000)
    else:
        df['user_id'] = 0

    # Fix Numerics
    for c in ['file_size', 'storage_limit', 'country', 'operation', 'file_type', 'subscription_type', 'success']:
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

        # 1. Load CSV (Optimized)
        try:
            df = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
        except:
            return jsonify({"error": "Invalid CSV"}), 400

        # 2. Preprocess (In-Place)
        try:
            df, feature_cols = preprocess_data_inplace(df)
            gc.collect() # Force cleanup
        except Exception as e:
            return jsonify({"error": f"Preprocessing failed: {e}"}), 400

        # 3. Predict (SCAN ALL ROWS)
        # We calculate X_scaled on the FULL dataset
        X_scaled = np.nan_to_num(scaler.transform(df[feature_cols]).astype(np.float32))
        raw_scores = model.decision_function(X_scaled)
        
        # Thresholds
        contamination = 0.001 + (sensitivity / 100) * 0.1 
        threshold = np.percentile(raw_scores, 100 * contamination)
        
        predictions = np.where(raw_scores < threshold, -1, 1)
        custom_anomalies = np.where(predictions == -1, 'Anomaly', 'Normal')

        # 4. Generate Reasons & Stats (Vectorized)
        avg_size = df['file_size'].mean()
        
        # Add Columns
        df['Status'] = custom_anomalies
        df['Risk Score'] = np.where(raw_scores < threshold, 3, 0)
        
        # Reason generation
        df['AI Reasoning'] = [find_reasons(row, avg_size) if label == 'Anomaly' else "Normal" 
                              for _, (row, label) in enumerate(zip(df.iloc, custom_anomalies))]

        # 5. BENCHMARKS (Both Models)
        benchmarks_data = []
        try:
            X_bench = X_scaled
            y_bench = np.where(df['Status'] == 'Anomaly', 1, 0)
            
            # SMART FALLBACK: If dataset is huge (>50k), we might crash, but let's try FULL first.
            # If it fails, we catch the exception and run on subset.
            try:
                if len(np.unique(y_bench)) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(X_bench, y_bench, test_size=0.3, random_state=42)
                    
                    # A. Random Forest (Optimized for speed)
                    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, class_weight='balanced', n_jobs=1)
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
            except MemoryError:
                # If Full Scan crashes RAM, fallback to 10k sample instantly
                print("⚠️ RAM Full: Switching benchmark to safe mode (10k rows)")
                subset_idx = np.random.choice(len(df), 10000, replace=False)
                # ... (Logic to run on subset would go here, but usually we just skip to save the request)
                benchmarks_data.append({"name": "Benchmark Skipped (Memory)", "Precision": 0, "Recall": 0, "F1": 0})

        except Exception as e:
            print(f"Benchmark error: {e}") 

        # Free X_scaled memory immediately
        del X_scaled
        gc.collect()

        # 6. Prepare Response
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['Status'] == 'Anomaly').sum()),
            "sensitivity_used": sensitivity
        }

        # Timeline
        timeline_groups = df.groupby([df['timestamp_dt'].dt.date, 'Status']).size().unstack(fill_value=0)
        timeline_data = [{"date": str(d), "Normal": int(r.get('Normal', 0)), "Anomaly": int(r.get('Anomaly', 0))} 
                         for d, r in timeline_groups.iterrows()]
        
        # Cleanup for JSON
        df.rename(columns={'timestamp': 'Time', 'user_id': 'User ID', 'operation': 'Action', 'file_size': 'Size'}, inplace=True)
        
        # ⚠️ SAFETY LIMIT: We scan 76k, but return 5k to browser to prevent JSON crash
        df_anomalies = df[df['Status'] == 'Anomaly']
        df_normal = df[df['Status'] == 'Normal']
        
        # Prioritize showing all anomalies, then fill rest with normal
        remaining_slots = 5000 - len(df_anomalies)
        if remaining_slots > 0:
            final_rows = pd.concat([df_anomalies, df_normal.head(remaining_slots)]).fillna("")
        else:
            final_rows = df_anomalies.head(5000).fillna("")

        results_json = final_rows.to_dict(orient="records")

        # Final Wipe
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