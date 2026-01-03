from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib  # For fast loading
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import gc   # Garbage Collector
import time 

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
    print("⚠️  SYSTEM WARNING: 'model.joblib' not found. Please run 'python train_model.py' locally first!")
    model = None
    scaler = None

# ==========================================
# 🧠 HELPER: EXPLAINABILITY (Why is this an anomaly?)
# ==========================================
def find_reasons(row, df_stats):
    reasons = []
    
    # 1. Check File Size (if 5x bigger than average)
    # Note: We check if the value is > 0 to avoid division errors
    if row['file_size'] > 0 and row['file_size'] > (df_stats['avg_size'] * 5):
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
# 🛠️ HELPER: DATA PREPROCESSING (Your Robust Version)
# ==========================================
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

    # 2. FILL MISSING & FEATURE ENGINEERING
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    else:
        df['timestamp_dt'] = pd.Timestamp('2024-01-01')

    if 'signup_date' in df.columns:
        signup_dt = pd.to_datetime(df['signup_date'], errors='coerce').fillna(df['timestamp_dt'])
        df['account_age_days'] = (df['timestamp_dt'] - signup_dt).dt.days.fillna(0).astype('int16')
    else:
        df['account_age_days'] = np.random.randint(0, 365, size=len(df)) # Simulated if missing

    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek.fillna(0).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype('int8')

    if 'success' in df.columns:
        # Robust check for strings "true", "success", "1"
        df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes', 'success']).astype('int8')
    else:
        df['success'] = 1

    # Ensure all required numeric features exist
    req_cols = ['file_size', 'storage_limit', 'subscription_type', 'country', 'operation', 'user_id', 'file_type', 'hour']
    for c in req_cols:
        if c not in df.columns:
            df[c] = 0
        else:
            # Force numeric for the Model
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
    start_time = time.time()

    if model is None or scaler is None:
        return jsonify({"error": "Server Error: Model not loaded. Please contact admin."}), 500

    try:
        if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        
        # --- 1. GET SENSITIVITY FROM SLIDER (New Feature) ---
        # 0 = Relaxed, 50 = Normal, 100 = Paranoid
        user_sensitivity = int(request.form.get('sensitivity', 50))
        
        # Math: Map 0-100 to Isolation Forest Threshold (-0.20 to -0.05)
        # Higher sensitivity = Closer to 0 (Easier to trigger anomaly)
        base_threshold = -0.13 
        adjustment = (user_sensitivity - 50) * 0.002 
        final_threshold = base_threshold + adjustment

        try:
            # Read CSV
            df_original = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
        except Exception as e:
            return jsonify({"error": f"CSV Read Error: Corrupted file."}), 400

        # --- 2. SECURITY BOUNCER (Your Robust Validation) ---
        df_original.columns = [c.lower().strip() for c in df_original.columns]
        required_concepts = {
            'User Identity': ['user', 'id', 'client', 'account', 'username'],
            'Data Size': ['size', 'bytes', 'length', 'storage'],
            'Timestamp': ['time', 'date', 'timestamp', 'created']
        }
        missing = [concept for concept, keys in required_concepts.items() 
                  if not any(k in c for k in keys for c in df_original.columns)]
        
        if missing:
             return jsonify({"error": f"⛔ Validation Failed. Missing: {', '.join(missing)}"}), 400

        # --- 3. PREPROCESS ---
        try:
            df, feature_cols = preprocess_data(df_original)
        except Exception as e:
            return jsonify({"error": f"Processing Error: {str(e)}"}), 400

        if df.empty: return jsonify({"error": "Empty dataset produced."}), 400

        # --- 4. PREDICT & EXPLAIN (New Feature) ---
        X_scaled = scaler.transform(df[feature_cols]).astype(np.float32)
        X_scaled = np.nan_to_num(X_scaled)

        # Get raw scores
        raw_scores = model.decision_function(X_scaled)
        
        # Lists for new columns
        custom_anomalies = []
        risks = []
        reasons_column = []
        
        # Stats for Reasoning
        stats = {'avg_size': df['file_size'].mean()}

        #  - This logic applies the slider threshold
        for i, score in enumerate(raw_scores):
            # Use SLIDER threshold instead of default model
            if score < final_threshold:
                custom_anomalies.append('Anomaly') 
                
                # Risk Logic
                if score < (final_threshold - 0.10):
                    risks.append(3) # Critical
                elif score < (final_threshold - 0.05):
                    risks.append(2) # High
                else:
                    risks.append(1) # Moderate
                
                # Call Explainability Helper
                reasons_column.append(find_reasons(df.iloc[i], stats))
            else:
                custom_anomalies.append('Normal')
                risks.append(0)
                reasons_column.append("Normal Activity")

        df['anomaly_score'] = risks
        df['anomaly_label'] = custom_anomalies
        df['Analysis'] = reasons_column # The "Why" column

        # --- 5. BENCHMARKS (Auto-Tuning Added) ---
        benchmarks_data = []
        subset_size = min(len(df), 15000) 
        idx = np.random.choice(len(df), subset_size, replace=False)
        y_truth = np.where(np.array(custom_anomalies)[idx] == 'Anomaly', 1, 0)
        X_bench = X_scaled[idx]
        
        if len(np.unique(y_truth)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_bench, y_truth, test_size=0.3, random_state=42, stratify=y_truth
            )

            # AUTO-TUNING LOOP (New Feature)
            best_f1 = 0
            best_rf = None
            
            # Try different brain sizes
            for n in [50, 100, 150]:
                rf_cand = RandomForestClassifier(n_estimators=n, random_state=42, class_weight='balanced')
                rf_cand.fit(X_train, y_train)
                pred_cand = rf_cand.predict(X_test)
                _, _, f1, _ = precision_recall_fscore_support(y_test, pred_cand, average='binary', zero_division=0)
                if f1 >= best_f1:
                    best_f1 = f1
                    best_rf = rf_cand
            
            # Final Metrics for Best RF
            rf_pred = best_rf.predict(X_test)
            p, r, f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='binary', zero_division=0)
            
            benchmarks_data.append({
                "name": f"Random Forest (Auto-Tuned n={best_rf.n_estimators})", 
                "Precision": round(p, 2), "Recall": round(r, 2), "F1": round(f1, 2)
            })

            # Logistic Regression
            lr = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced')
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            p, r, f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='binary', zero_division=0)
            benchmarks_data.append({
                "name": "Logistic Regression", 
                "Precision": round(p, 2), "Recall": round(r, 2), "F1": round(f1, 2)
            })

        # --- 6. TIMELINE (Kept from your Robust Code) ---
        df['dt_temp'] = df['timestamp_dt']
        timeline_groups = df.groupby([df['dt_temp'].dt.date, 'anomaly_label']).size().unstack(fill_value=0)
        
        timeline_data = []
        for date, row in timeline_groups.iterrows():
            timeline_data.append({
                "date": str(date),
                "Normal": int(row.get('Normal', 0)),
                "Anomaly": int(row.get('Anomaly', 0))
            })
        timeline_data.sort(key=lambda x: x['date'])

        # --- 7. EXPORT PREPARATION (Mapping Back to Strings) ---
        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum()),
            "sensitivity_used": user_sensitivity
        }

        # Recover Readable Strings
        df['timestamp'] = df['timestamp_dt'].astype(str)
        country_map = {0: 'Other/Unknown', 1: 'FR', 2: 'GB', 3: 'PL', 4: 'UA', 5: 'US'}
        df['country'] = df['country'].map(country_map).fillna(df['country'])
        
        action_map = {0: 'delete', 1: 'download', 2: 'modify', 3: 'upload'}
        df['operation'] = df['operation'].map(action_map).fillna(df['operation'])
        
        file_map = {0: 'archive', 1: 'document', 2: 'photo', 3: 'video'}
        df['file_type'] = df['file_type'].map(file_map).fillna(df['file_type'])
        
        plan_map = {0: 'business', 1: 'free', 2: 'premium'}
        df['subscription_type'] = df['subscription_type'].map(plan_map).fillna(df['subscription_type'])
        
        success_map = {1: 'Success', 0: 'Failed'}
        df['success'] = df['success'].map(success_map).fillna(df['success'])

        df_export = df.copy()
        cols_to_drop = ['timestamp_dt', 'signup_date_dt', 'dt_temp'] 
        df_export.drop(columns=[c for c in cols_to_drop if c in df_export.columns], inplace=True, errors='ignore')

        pretty_names = {
            'user_id': 'User ID', 'anomaly_label': 'Status', 'anomaly_score': 'Risk Score',
            'timestamp': 'Time', 'country': 'Country', 'operation': 'Action',
            'file_type': 'File Type', 'file_size': 'Size', 'subscription_type': 'Plan',
            'success': 'Success', 'account_age_days': 'Account Age', 
            'Analysis': 'AI Reasoning' # <-- The new column name
        }
        df_export.rename(columns=pretty_names, inplace=True)

        # Cleanup
        del df, X_scaled, df_original
        gc.collect()

        # Limit Rows
        MAX_ROWS = 5000
        df_anomalies = df_export[df_export['Status'] == 'Anomaly']
        remaining = MAX_ROWS - len(df_anomalies)
        if remaining > 0:
            df_normal = df_export[df_export['Status'] == 'Normal'].head(remaining)
            df_final = pd.concat([df_anomalies, df_normal])
        else:
            df_final = df_anomalies.head(MAX_ROWS)
            
        return jsonify({
            "summary": summary,
            "benchmarks": benchmarks_data,
            "timeline": timeline_data,
            "results": df_final.fillna("").to_dict(orient="records")
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)