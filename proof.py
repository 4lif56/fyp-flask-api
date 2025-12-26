import time
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION ---
FILE_NAME = 'cleaned_data.csv'  # Make sure this matches your file name exactly!

print(f"Attempting to load {FILE_NAME}...")

try:
    # --- 2. LOAD DATA ---
    # We read the file just like your app does
    df = pd.read_csv(FILE_NAME)
    print(f"✅ Successfully loaded {len(df)} rows.")

    # --- 3. PREPARE FEATURES ---
    # Select only numeric columns (since Isolation Forest needs numbers)
    # This mimics your 'feature engineering' step
    data_to_fit = df.select_dtypes(include=['float64', 'int64', 'int32', 'int16', 'int8'])
    
    # Fill any NaNs just in case (safety check)
    data_to_fit = data_to_fit.fillna(0)

    # --- 4. START THE SPEED TEST ⏱️ ---
    print("-" * 30)
    print("🚀 Starting Algorithm Speed Test...")
    start_time = time.time()

    # Step A: Scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_to_fit)

    # Step B: Isolation Forest Training & Prediction
    # (Using the same settings as your thesis: 100 estimators)
    model = IsolationForest(n_estimators=100, n_jobs=-1, contamination=0.01)
    model.fit(X_scaled)
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    end_time = time.time()
    # ------------------------------------

    duration = round(end_time - start_time, 4)
    
    print("-" * 30)
    print(f"✅ FINAL RESULT:")
    print(f"Processed {len(df)} rows in: {duration} seconds")
    print("-" * 30)

except FileNotFoundError:
    print(f"❌ ERROR: Could not find '{FILE_NAME}' in this folder.")
    print("Please check the file name and try again.")

except Exception as e:
    print(f"❌ ERROR: Something went wrong.\n{e}")