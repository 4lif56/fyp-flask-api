from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import gc
import time
from io import BytesIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# ==============================
# HELPER: FORMAT BYTES (Now with TB!)
# ==============================
def human_readable_size(bytes_val):
    try:
        bytes_val = float(bytes_val)
        if bytes_val >= 1024**4:
            return f"{bytes_val / (1024**4):.2f} TB"
        elif bytes_val >= 1024**3:
            return f"{bytes_val / (1024**3):.2f} GB"
        elif bytes_val >= 1024**2:
            return f"{bytes_val / (1024**2):.2f} MB"
        elif bytes_val >= 1024:
            return f"{bytes_val / 1024:.2f} KB"
        else:
            return f"{bytes_val:.0f} B"
    except:
        return ""

# ==============================
# CONFIG (Render-safe)
# ==============================
MAX_OUTPUT_ROWS = 5000
MAX_BENCH_ROWS = 15000
DEFAULT_SENSITIVITY = 50

app = Flask(__name__)
CORS(app)
app.json.sort_keys = False

# ==============================
# LOAD MODEL ONCE
# ==============================
print("⏳ Loading model...")
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model, scaler = None, None

# ==============================
# EXPLAINABILITY
# ==============================
def find_reasons(row, stats):
    reasons = []
    # Use .get() for safety
    val_size = row.get("file_size", 0)
    val_hour = row.get("hour", 0)
    val_age = row.get("account_age_days", 0)
    val_success = row.get("success", 1)

    if val_size > 0 and val_size > stats["avg_size"] * 5:
        reasons.append("Unusual File Size")

    if 3 <= val_hour <= 5:
        reasons.append("Odd Hours (Night)")

    if val_age < 1:
        reasons.append("New Account")

    if val_success == 0:
        reasons.append("Failed Operation")

    return ", ".join(reasons) if reasons else "Pattern Deviation"

# ==============================
# PREPROCESSING (FUZZY MATCHING)
# ==============================
def preprocess_data(df):
    rename_map = {
        "timestamp": ["date", "time", "created_at", "datetime"],
        "user_id": ["id", "client_id", "customer_id", "account_id", "username"],
        "file_size": ["size", "bytes", "length"],
        "operation": ["action", "activity", "event", "type"],
        "success": ["status", "is_success", "result"],
        "country": ["region", "location", "geo"],
        "file_type": ["file_format", "extension"],
        "subscription_type": ["plan", "tier"],
    }

    curr_cols = set(df.columns)
    rename = {}

    for std, alts in rename_map.items():
        if std not in curr_cols:
            for alt in alts:
                match = next((c for c in curr_cols if c.lower() == alt), None)
                if match:
                    rename[match] = std
                    break

    if rename:
        df.rename(columns=rename, inplace=True)

    # Timestamp
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(
            pd.Timestamp("2024-01-01")
        )
    else:
        df["timestamp_dt"] = pd.Timestamp("2024-01-01")

    # Account age
    if "signup_date" in df.columns:
        signup_dt = pd.to_datetime(df["signup_date"], errors="coerce").fillna(
            df["timestamp_dt"]
        )
        df["account_age_days"] = (
            df["timestamp_dt"] - signup_dt
        ).dt.days.fillna(0).astype("int16")
    else:
        df["account_age_days"] = np.random.randint(0, 365, size=len(df))

    df["day_of_week"] = df["timestamp_dt"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["hour"] = df["timestamp_dt"].dt.hour.astype("int8")

    if "success" in df.columns:
        df["success"] = (
            df["success"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "success"])
            .astype("int8")
        )
    else:
        df["success"] = 1

    required = [
        "file_size",
        "storage_limit",
        "subscription_type",
        "country",
        "operation",
        "user_id",
        "file_type",
        "hour",
    ]

    for col in required:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    features = [
        "user_id",
        "operation",
        "file_type",
        "file_size",
        "success",
        "subscription_type",
        "storage_limit",
        "country",
        "hour",
        "account_age_days",
        "day_of_week",
        "is_weekend",
    ]

    return df, features

# ==============================
# MAIN ROUTE
# ==============================
@app.route("/detect", methods=["POST"])
def detect():
    start = time.time()

    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        sensitivity = float(request.form.get("sensitivity", DEFAULT_SENSITIVITY))
        df_raw = pd.read_csv(request.files["file"], encoding="utf-8-sig", low_memory=False)
        df_raw.columns = [c.lower().strip() for c in df_raw.columns]

        # Security validation
        concepts = {
            "User Identity": ["user", "id", "account"],
            "Data Size": ["size", "bytes", "storage"],
            "Timestamp": ["time", "date"],
        }

        missing = [
            name
            for name, keys in concepts.items()
            if not any(k in c for c in df_raw.columns for k in keys)
        ]

        if missing:
            return jsonify({"error": f"Missing: {', '.join(missing)}"}), 400

        df, feature_cols = preprocess_data(df_raw)

        X = scaler.transform(df[feature_cols]).astype(np.float32)
        X = np.nan_to_num(X)

        scores = model.decision_function(X)

        contamination = 0.001 + (sensitivity / 100) * 0.1
        threshold = np.percentile(scores, 100 * contamination)

        preds = np.where(scores < threshold, -1, 1)

        critical = np.percentile(scores, 100 * contamination * 0.2)

        def risk(score, pred):
            if pred == 1:
                return 0
            if score < critical:
                return 3
            return 2

        df["anomaly_score"] = np.vectorize(risk)(scores, preds)
        df["anomaly_label"] = np.where(preds == -1, "Anomaly", "Normal")

        stats = {"avg_size": df["file_size"].mean()}
        df["Analysis"] = [
            find_reasons(df.iloc[i], stats) if p == "Anomaly" else "Normal Activity"
            for i, p in enumerate(df["anomaly_label"])
        ]

        # ==========================
        # BENCHMARKING (SAFE)
        # ==========================
        benchmarks = []
        subset = min(len(df), MAX_BENCH_ROWS)
        idx = np.random.choice(len(df), subset, replace=False)
        
        df_sample = df.iloc[idx].copy()
        Xb = X[idx]
        y = (df_sample["anomaly_label"] == "Anomaly").astype(int)

        unique_y, counts = np.unique(y, return_counts=True)
        # Safe split: Needs at least 2 classes and 2 items per class
        if len(unique_y) > 1 and np.min(counts) >= 2:
            Xtr, Xte, ytr, yte = train_test_split(
                Xb, y, test_size=0.3, stratify=y, random_state=42
            )
        else:
            Xtr = None

        if Xtr is not None:
            rf_best, best_f1 = None, 0
            for n in (50, 100, 150):
                rf = RandomForestClassifier(
                    n_estimators=n, random_state=42, class_weight="balanced"
                )
                rf.fit(Xtr, ytr)
                _, _, f1, _ = precision_recall_fscore_support(
                    yte, rf.predict(Xte), average="binary", zero_division=0
                )
                if f1 >= best_f1:
                    best_f1, rf_best = f1, rf

            p, r, f1, _ = precision_recall_fscore_support(
                yte, rf_best.predict(Xte), average="binary", zero_division=0
            )
            benchmarks.append(
                {
                    "name": f"Random Forest (n={rf_best.n_estimators})",
                    "Precision": round(p, 2),
                    "Recall": round(r, 2),
                    "F1": round(f1, 2),
                }
            )

            lr = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced")
            lr.fit(Xtr, ytr)
            p, r, f1, _ = precision_recall_fscore_support(
                yte, lr.predict(Xte), average="binary", zero_division=0
            )
            benchmarks.append(
                {"name": "Logistic Regression", "Precision": round(p, 2), "Recall": round(r, 2), "F1": round(f1, 2)}
            )

        # ==========================
        # TIMELINE (FIXED: CONTINUOUS DATES)
        # ==========================
        df["date_only"] = df["timestamp_dt"].dt.floor("D")

        timeline_df = (
            df.groupby(["date_only", "anomaly_label"])
            .size()
            .unstack(fill_value=0)
        )

        if not timeline_df.empty:
            full_range = pd.date_range(
                start=timeline_df.index.min(), 
                end=timeline_df.index.max(), 
                freq="D"
            )
            timeline_df = timeline_df.reindex(full_range, fill_value=0)
            timeline_df.index.name = "date_only"

        timeline = [
            {
                "date": date_val.strftime("%Y-%m-%d"),
                "Normal": int(row.get("Normal", 0)),
                "Anomaly": int(row.get("Anomaly", 0)),
            }
            for date_val, row in timeline_df.iterrows()
        ]

        # ==========================
        # EXPORT & MAPPING (THE FIX)
        # ==========================
        summary = {
            "total_rows": len(df),
            "anomalies": int((df["anomaly_label"] == "Anomaly").sum()),
            "sensitivity_used": sensitivity,
        }

        df["timestamp"] = df["timestamp_dt"].astype(str)
        df.drop(columns=["timestamp_dt", "date_only"], inplace=True, errors="ignore")

        # --- MAPPINGS (Numbers -> Readable Strings) ---
        # 1. Subscription Type (0=Business, 1=Free, 2=Premium)
        sub_map = {0: 'Business', 1: 'Free', 2: 'Premium'}
        df['subscription_type'] = df['subscription_type'].map(sub_map).fillna('Unknown')

        # 2. Operation (0=Upload, 1=Download, 2=Modify, 3=Delete)
        op_map = {0: 'Upload', 1: 'Download', 2: 'Modify', 3: 'Delete'}
        df['operation'] = df['operation'].map(op_map).fillna('Other')

        # 3. File Type (1=Document, 2=Photo, 3=Video)
        type_map = {0: 'Archive', 1: 'Document', 2: 'Photo', 3: 'Video'}
        df['file_type'] = df['file_type'].map(type_map).fillna('File')

        # 4. Success (1=True, 0=False)
        success_map = {1: 'True', 0: 'False'}
        df['success'] = df['success'].map(success_map).fillna('False')
        
        # ---------------------------------------------

        df_out = df.rename(
            columns={
                "user_id": "User ID",
                "timestamp": "Time",
                "anomaly_label": "Status",
                "anomaly_score": "Risk Score",
                "file_size": "Size",
                "storage_limit": "Limit", 
                "Analysis": "AI Reasoning",
                "subscription_type": "Plan",
                "operation": "Action",
                "file_type": "File Type",
                "success": "Success"
            }
        )

        # Apply formatting to BOTH Size and Limit
        if "Size" in df_out.columns:
            df_out["Size"] = df_out["Size"].apply(human_readable_size)
        
        if "Limit" in df_out.columns:
            df_out["Limit"] = df_out["Limit"].apply(human_readable_size)

        anomalies = df_out[df_out["Status"] == "Anomaly"]
        normal = df_out[df_out["Status"] == "Normal"]

        if len(anomalies) < MAX_OUTPUT_ROWS:
            df_out = pd.concat([anomalies, normal.head(MAX_OUTPUT_ROWS - len(anomalies))])
        else:
            df_out = anomalies.head(MAX_OUTPUT_ROWS)

        del df, df_raw, X
        gc.collect()

        return jsonify(
            {
                "summary": summary,
                "benchmarks": benchmarks,
                "timeline": timeline,
                "results": df_out.fillna("").to_dict("records"),
                "runtime_sec": round(time.time() - start, 2),
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==========================================
# 📥 DOWNLOAD TEMPLATE (Fixed: 20 Rows)
# ==========================================
@app.route('/template', methods=['GET'])
def download_template():
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
    app.run(host="0.0.0.0", port=5000)