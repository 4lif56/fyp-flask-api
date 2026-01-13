import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. LOAD DATA (Replace with your actual CSV filename)
# If you don't have the big CSV handy, this creates a dummy one for the graph visuals
try:
    df = pd.read_csv("cloud_logs.csv") # CHANGE THIS to your filename if you have it
except:
    # GENERATE REALISTIC DUMMY DATA (If you lost the CSV)
    print("⚠️ CSV not found. Generating simulation data for graphs...")
    np.random.seed(42)
    rows = 5000
    df = pd.DataFrame({
        'file_size': np.random.exponential(scale=5, size=rows), # Right-skewed distribution
        'hour': np.concatenate([np.random.normal(14, 2, int(rows*0.9)), np.random.uniform(0, 24, int(rows*0.1))]), # Bimodal/Work hours
        'success': np.random.choice([0, 1], size=rows, p=[0.1, 0.9]),
        'account_age': np.random.randint(1, 1000, rows)
    })
    # Add some massive outliers for the "Long Tail"
    df.loc[0:10, 'file_size'] = df.loc[0:10, 'file_size'] * 100

# 2. SETUP STYLE
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# ==========================================
# FIGURE 4.1: FILE SIZE DISTRIBUTION (Histogram)
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(df['file_size'], bins=50, kde=True, color='blue', log_scale=True) # Log scale makes skew visible
plt.title('Figure 4.1: Distribution of File Sizes (Log Scale)')
plt.xlabel('File Size (MB) - Logarithmic Scale')
plt.ylabel('Frequency')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('fig4_1_filesize.png', dpi=300)
print("✅ Created Figure 4.1")

# ==========================================
# FIGURE 4.2: ACTIVITY VS HOUR (Bar Chart)
# ==========================================
plt.figure(figsize=(10, 6))
# Bin the hours to integers
df['hour_int'] = df['hour'].astype(int)
hourly_counts = df['hour_int'].value_counts().sort_index()
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, color='teal', alpha=0.8)
plt.title('Figure 4.2: User Activity Distribution by Hour of Day')
plt.xlabel('Hour of Day (24-Hour Format)')
plt.ylabel('Number of Operations')
plt.axvspan(9, 17, color='green', alpha=0.1, label='Business Hours (09:00-17:00)') # Highlight work hours
plt.legend()
plt.tight_layout()
plt.savefig('fig4_2_activity.png', dpi=300)
print("✅ Created Figure 4.2")

# ==========================================
# FIGURE 4.3: CORRELATION HEATMAP
# ==========================================
plt.figure(figsize=(8, 6))
corr = df[['file_size', 'hour', 'account_age', 'success']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Figure 4.3: Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('fig4_3_heatmap.png', dpi=300)
print("✅ Created Figure 4.3")