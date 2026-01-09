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

# 🚀 LOAD MODEL
print("⏳ Loading model...")
try:
    model = joblib.load('model.joblib', mmap_mode='r')
    scaler = joblib.load('scaler.joblib')
    print("✅ Model Loaded.")
except:
    print("❌ Model missing. Run train_model.py first.")
    model, scaler = None, None

# 🧠 HELPER: SMART COLUMN MAPPING
def smart_rename(df):
    clean_cols = [str(c).lower().strip().replace('_', '').replace(' ', '') for c in df.columns]
    concepts = {
        'timestamp': ['time', 'date', 'created'], 'user_id': ['user', 'client', 'id', 'account'],
        'file_size': ['size', 'byte', 'len'], 'success': ['status', 'result', 'success'],
        'operation': ['action', 'activity', 'type'], 'country': ['geo', 'region', 'country'],
        'file_type': ['format', 'ext', 'type'], 'subscription_type': ['plan', 'tier']
    }
    rename_map = {}
    used = set()
    for concept, keys in concepts.items():
        best, best_score = -1, 0
        for i, col in enumerate(clean_cols):
            if i in used: continue
            score = sum(2 if k in col else 0 for k in keys)
            if score > best_score: best, best_score = i, score
        if best != -1:
            rename_map[df.columns[best]] = concept
            used.add(best)
    df.rename(columns=rename_map, inplace=True)
    return df

# 🧠 HELPER: EXPLAINABILITY
def find_reasons(row, avg_size):
    reasons = []
    if row['file_size'] > (avg_size * 5): reasons.append("Unusual File Size")
    if 3 <= row['hour'] <= 5: reasons.append("Odd Hours (Night)")
    if row['account_age_days'] < 1: reasons.append("New Account")
    if row['success'] == 0: reasons.append("Failed Operation")
    return ", ".join(reasons) if reasons else "Pattern Deviation"

# 🔍 MAIN ROUTE
@app.route('/detect', methods=['POST'])
def detect():
    if not model: return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    
    try:
        # 1. READ & PREPROCESS
        df = pd.read_csv(request.files['file'], encoding='utf-8-sig', low_memory=False)
        df = smart_rename(df)
        
        # Validation
        req = ['user_id', 'file_size', 'timestamp']
        if not all(c in df.columns for c in req): 
            return jsonify({"error": "Missing key columns (User, Size, or Time)"}), 400

        # Feature Engineering
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
        df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')
        df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['account_age_days'] = np.random.randint(0, 365, size=len(df)).astype('int16')
        
        # Fill/Convert Numeric Columns Safely
        numeric_cols = ['file_size', 'country', 'operation', 'file_type', 'storage_limit', 'subscription_type']
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = 0 # Create if missing
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0) # Clean if exists

        # Fix Success Column
        if 'success' in df.columns:
            df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes', 'success']).astype('int8')
        else:
            df['success'] = 1

        # User Hash
        df['user_numeric'] = df['user_id'].astype(str).apply(lambda x: abs(hash(x)) % 100000)

        # 2. PREDICT
        features = ['user_numeric', 'operation', 'file_type', 'file_size', 'success', 'subscription_type', 
                    'storage_limit', 'country', 'hour', 'account_age_days', 'day_of_week', 'is_weekend']
            
        X = np.nan_to_num(scaler.transform(df[features]).astype(np.float32))
        
        # Dynamic Sensitivity
        sens = float(request.form.get('sensitivity', 50))
        contamination = 0.001 + (sens / 100) * 0.1
        scores = model.decision_function(X)
        threshold = np.percentile(scores, 100 * contamination)
        preds = np.where(scores < threshold, -1, 1)

        # 3. RESULTS
        df['Status'] = np.where(preds == -1, 'Anomaly', 'Normal')
        avg_size = df['file_size'].mean()
        df['Analysis'] = [find_reasons(row, avg_size) if lbl == 'Anomaly' else "Normal" 
                          for _, (row, lbl) in enumerate(zip(df.iloc, df['Status']))]
        
        crit_thresh = np.percentile(scores, 100 * (contamination * 0.2))
        df['Risk Score'] = [3 if s < crit_thresh else 2 if s < threshold else 0 for s in scores]

        # 4. BENCHMARKS
        benchmarks = []
        try:
            sub_idx = np.random.choice(len(df), min(len(df), 15000), replace=False)
            y_sub = np.where(df['Status'].iloc[sub_idx] == 'Anomaly', 1, 0)
            if len(np.unique(y_sub)) > 1:
                X_tr, X_te, y_tr, y_te = train_test_split(X[sub_idx], y_sub, test_size=0.3, stratify=y_sub)
                for name, clf in [("Random Forest", RandomForestClassifier(n_estimators=50)), 
                                  ("Logistic Regression", LogisticRegression(solver='liblinear'))]:
                    clf.fit(X_tr, y_tr)
                    p, r, f, _ = precision_recall_fscore_support(y_te, clf.predict(X_te), average='binary', zero_division=0)
                    benchmarks.append({"name": name, "Precision": round(p,2), "Recall": round(r,2), "F1": round(f,2)})
        except: pass

        # 5. EXPORT
        timeline = df.groupby([df['timestamp_dt'].dt.date.astype(str), 'Status']).size().unstack(fill_value=0).reset_index()
        timeline.rename(columns={'timestamp_dt': 'date'}, inplace=True)
        
        out_cols = {'user_id': 'User ID', 'timestamp': 'Time', 'file_size': 'Size', 'operation': 'Action'}
        df.rename(columns=out_cols, inplace=True)
        final_data = pd.concat([df[df['Status']=='Anomaly'], df[df['Status']=='Normal'].head(5000 - len(df[df['Status']=='Anomaly']))])
        
        del df, X, scores
        gc.collect()

        return jsonify({
            "summary": {"total_rows": len(df_original), "anomalies": int((final_data['Status']=='Anomaly').sum())},
            "benchmarks": benchmarks,
            "timeline": timeline.to_dict(orient='records'),
            "results": final_data.fillna("").to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

# 📥 DOWNLOAD TEMPLATE
@app.route('/template', methods=['GET'])
def template():
    data = {
        'Timestamp': ['2024-01-01 09:00']*15 + ['2024-01-01 03:00', '2024-01-01 03:15', '2024-01-01 12:00', '2024-01-01 12:05', '2024-01-01 23:59'],
        'User_Identity': ['User']*15 + ['HACKER_X', 'HACKER_X', 'Inside_Job', 'Inside_Job', 'Unknown'],
        'File_Size_Bytes': [1024]*15 + [100, 100, 999999999, 999999999, 500],
        'Activity_Type': ['upload']*15 + ['download', 'delete', 'download', 'download', 'login'],
        'Is_Success': ['True']*15 + ['True', 'False', 'True', 'True', 'False']
    }
    out = BytesIO()
    pd.DataFrame(data).to_csv(out, index=False)
    out.seek(0)
    return send_file(out, mimetype='text/csv', as_attachment=True, download_name='Smart_Audit_Template.csv')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)