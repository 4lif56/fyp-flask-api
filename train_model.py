import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib  # This is the secret weapon for speed

# 1. Load your data
print("Loading data...")
df = pd.read_csv('cleaned_data.csv')

# 2. Select the features (Make sure these match what you use in app.py!)
# Based on your CSV, these seem to be the numeric columns you use:
features = ['user_id', 'operation', 'file_type', 'file_size', 'subscription_type', 'storage_limit', 'country', 'hour']

# Filter only the columns we need
data = df[features]
data = data.fillna(0) # Safety check

# 3. Scale the data (Optional but recommended, if you use it in app.py keep it)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 4. Train the Model
print("Training Isolation Forest... (This might take a moment)")
model = IsolationForest(n_estimators=100, contamination=0.05, n_jobs=-1, random_state=42)
model.fit(X_scaled)

# 5. Save the "Brain" using Joblib (Compressed)
print("Saving model...")
joblib.dump(model, 'model.joblib', compress=3)
joblib.dump(scaler, 'scaler.joblib', compress=3) # Save the scaler too if you use it!

print("✅ Done! You now have 'model.joblib' and 'scaler.joblib'.")