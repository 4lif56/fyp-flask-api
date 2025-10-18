from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
import io

app = Flask(__name__)
CORS(app)


@app.route('/detect', methods=['POST'])
def detect_anomalies():
    file = request.files.get('file')
    print("📂 Received file:", file)


    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Read CSV file
    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    # Check numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return jsonify({"error": "No numeric columns found"}), 400

    # Apply Isolation Forest
    model = IsolationForest(contamination=0.02, random_state=42)
    df["anomaly_label"] = model.fit_predict(df[numeric_cols])
    df["anomaly_label"] = df["anomaly_label"].map({1: "Normal", -1: "Anomaly"})

    # Count summary
    total = len(df)
    anomalies = len(df[df["anomaly_label"] == "Anomaly"])

    return jsonify({
        "summary": {"total_rows": total, "anomalies": anomalies},
        "results": df.head(50).to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)
