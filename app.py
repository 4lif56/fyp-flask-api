from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import time

app = Flask(__name__)
CORS(app)


@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Record start time
        start_time = time.perf_counter()

        # Run Isolation Forest
        iso = IsolationForest(contamination=0.02, random_state=42)
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        df['anomaly_score'] = iso.fit_predict(numeric_df)
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # Count anomalies
        anomalies_count = int((df['anomaly_label'] == 'Anomaly').sum())

        # Calculate processing duration
        duration = round(time.perf_counter() - start_time, 3)

        # Initialize default metric values
        accuracy = precision = recall = fpr = None

        # If dataset includes an 'actual_label' column, calculate metrics
        if 'actual_label' in df.columns:
            y_true = df['actual_label'].map({'Normal': 0, 'Anomaly': 1})
            y_pred = df['anomaly_label'].map({'Normal': 0, 'Anomaly': 1})

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
            precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
            recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
            fpr = round(fp / (fp + tn) * 100, 2)

        # Build summary dictionary
        summary = {
            "total_rows": len(df),
            "anomalies": anomalies_count,
            "duration": f"{duration}s",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "fpr": fpr
        }

        # Clean dataset (handle NaN / infinity)
        df = df.fillna("")
        df = df.replace([float('inf'), float('-inf')], "")

        return jsonify({
            "summary": summary,
            "results": df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
