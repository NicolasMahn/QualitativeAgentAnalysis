import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('uploads/unclean_smartwatch_health_data.csv')

# 1. Handle Missing Values
# Drop rows with missing User ID
df = df.dropna(subset=['User ID'])
# Convert User ID to integer
df['User ID'] = df['User ID'].astype(int)

# Impute numeric columns with median
numeric_cols = ['Heart Rate (BPM)', 'Blood Oxygen Level (%)', 'Step Count']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fix Sleep Duration: Replace 'ERROR' and convert to float
df['Sleep Duration (hours)'] = pd.to_numeric(df['Sleep Duration (hours)'], errors='coerce')
df['Sleep Duration (hours)'] = df['Sleep Duration (hours)'].fillna(df['Sleep Duration (hours)'].median())

# Standardize Activity Level
df['Activity Level'] = df['Activity Level'].str.replace('_', ' ').str.replace('Actve', 'Active')
df['Activity Level'] = df['Activity Level'].fillna(df['Activity Level'].mode()[0])

# Clean Stress Level: Convert to numeric, replace invalid entries, impute with mode
df['Stress Level'] = pd.to_numeric(df['Stress Level'], errors='coerce')
df['Stress Level'] = df['Stress Level'].clip(1, 5)  # Assume valid range is 1-5
df['Stress Level'] = df['Stress Level'].fillna(df['Stress Level'].mode()[0]).astype(int)

# 2. Handle Outliers
# Heart Rate: 30-200 BPM
df['Heart Rate (BPM)'] = df['Heart Rate (BPM)'].clip(30, 200)
# Blood Oxygen: 90-100%
df['Blood Oxygen Level (%)'] = df['Blood Oxygen Level (%)'].clip(90, 100)
# Step Count: Cap at 30k
df['Step Count'] = df['Step Count'].clip(upper=30000)
# Sleep Duration: 0-24 hours
df['Sleep Duration (hours)'] = df['Sleep Duration (hours)'].clip(0, 24)

# 3. Remove duplicates
df = df.drop_duplicates()

# Save cleaned data
df.to_csv('cleaned_smartwatch_health_data.csv', index=False)