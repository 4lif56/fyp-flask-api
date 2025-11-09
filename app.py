from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import io

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- This is the new Preprocessing Function ---
# We integrated the logic from our evaluation script here
def preprocess_data(df):
    """
    Cleans and prepares the uploaded CSV file for machine learning.
    """
    # We must drop rows where 'timestamp' or 'hour' is missing
    # because we cannot process them.
    df = df.dropna(subset=['timestamp', 'hour'])

    # 1. Convert timestamp and signup_date to datetime objects
    try:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    except ValueError:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp']) # Try default parser

    try:
        df['signup_date_dt'] = pd.to_datetime(df['signup_date'], format='%d/%m/%Y')
    except ValueError:
        df['signup_date_dt'] = pd.to_datetime(df['signup_date']) # Try default parser

    # 2. Feature Engineering: Create new, useful features
    df['account_age_days'] = (df['timestamp_dt'] - df['signup_date_dt']).dt.days
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3. Convert boolean 'success' to integer
    df['success'] = df['success'].apply(lambda x: 1 if x else 0)

    # 4. Define feature columns to be used by the model
    feature_columns = [
        'user_id', 
        'operation', 
        'file_type', 
        'file_size', 
        'success',
        'subscription_type', 
        'storage_limit', 
        'country', 
        'hour',
        'account_age_days',
        'day_of_week',
        'is_weekend'
    ]
    
    # Ensure all data is numeric and handle any potential infinities
    final_df = df.copy() # Use .copy() to avoid SettingWithCopyWarning
    final_df[feature_columns] = final_df[feature_columns].replace([np.inf, -np.inf], np.nan)
    final_df[feature_columns] = final_df[feature_columns].fillna(0) # Fill any remaining NaNs
    
    return final_df, feature_columns

# --- Your Updated API Endpoint ---
@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
            
        df_original = pd.read_csv(file)
        
        # --- 1. PREPROCESS THE DATA ---
        df, feature_columns = preprocess_data(df_original.copy())

        if df.empty:
            return jsonify({"error": "No valid data to process after cleaning."}), 400

        # --- 2. SCALE THE FEATURES ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_columns])

        # --- 3. RUN ISOLATION FOREST ---
        # We use contamination=0.01 as determined in our evaluation
        iso = IsolationForest(contamination=0.01, random_state=42)
        
        # Fit and predict on the SCALED data
        predictions = iso.fit_predict(X_scaled)

        # Add results back to the *original* dataframe
        df['anomaly_score'] = predictions
        df['anomaly_label'] = df['anomaly_score'].map({1: 'Normal', -1: 'Anomaly'})

        # --- 4. PREPARE JSON RESPONSE ---
        summary = {
            "total_rows_processed": len(df),
            "anomalies_found": int((df['anomaly_label'] == 'Anomaly').sum())
        }
        
        # Clean up the dataframe for JSON conversion
        # Replace NaN/Inf with strings so it doesn't break JSON
        df = df.fillna("").replace([np.inf, -np.inf], "")
        # Convert datetime columns to string
        df['timestamp_dt'] = df['timestamp_dt'].astype(str)
        df['signup_date_dt'] = df['signup_date_dt'].astype(str)

        return jsonify({
            "summary": summary,
            "results": df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)