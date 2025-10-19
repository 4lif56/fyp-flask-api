from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
import io

app = Flask(__name__)
CORS(app)


@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Run Isolation Forest (your anomaly detection logic)
        iso = IsolationForest(contamination=0.02, random_state=42)
        df['anomaly_score'] = iso.fit_predict(df.select_dtypes(include=['float64', 'int64']))
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        summary = {
            "total_rows": len(df),
            "anomalies": int((df['anomaly_label'] == 'Anomaly').sum())
        }

        df = df.fillna("")  # Replace NaN with empty string before sending

        return jsonify({
            "summary": summary,
            "results": df.to_dict(orient="records")
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
