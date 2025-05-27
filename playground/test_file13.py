import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('uploads/Employee.csv')

# Exploratory Data Analysis
print("=== Dataset Information ===")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nValue counts for categorical columns:")
print("Education:", df['Education'].value_counts(), sep='\n')
print("\nCity:", df['City'].value_counts(), sep='\n')
print("\nGender:", df['Gender'].value_counts(), sep='\n')
print("\nEverBenched:", df['EverBenched'].value_counts(), sep='\n')

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='Education', hue='LeaveOrNot', data=df)
plt.title('Education Level vs Leave Status')

plt.subplot(2, 2, 2)
sns.boxplot(x='LeaveOrNot', y='Age', data=df)
plt.title('Age Distribution by Leave Status')

plt.subplot(2, 2, 3)
sns.countplot(x='EverBenched', hue='LeaveOrNot', data=df)
plt.title('Bench History vs Leave Status')

plt.subplot(2, 2, 4)
sns.countplot(x='PaymentTier', hue='LeaveOrNot', data=df)
plt.title('Payment Tier vs Leave Status')

plt.tight_layout()
plt.show()

# Preprocessing
X = df.drop('LeaveOrNot', axis=1)
y = df['LeaveOrNot']

# Define preprocessing steps
categorical_features = ['Education', 'City', 'Gender', 'EverBenched']
numeric_features = ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Model pipeline
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n{model.__class__.__name__} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return pipeline

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_pipeline = train_evaluate_model(lr_model, X_train, X_test, y_train, y_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pipeline = train_evaluate_model(rf_model, X_train, X_test, y_train, y_test)

# Feature Importance for Random Forest
encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
feature_names = numeric_features + list(encoder.get_feature_names_out(categorical_features))

importances = rf_pipeline.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar')
plt.title('Feature Importances from Random Forest')
plt.ylabel('Importance Score')
plt.show()