import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('uploads/Employee.csv')

# Exploratory Data Analysis
print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Visualize distribution of key features
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.countplot(x='Education', data=df)
plt.title('Education Distribution')

plt.subplot(2, 2, 2)
sns.histplot(df['Age'], bins=20)
plt.title('Age Distribution')

plt.subplot(2, 2, 3)
sns.countplot(x='PaymentTier', data=df)
plt.title('Payment Tier Distribution')

plt.subplot(2, 2, 4)
sns.countplot(x='LeaveOrNot', data=df)
plt.title('Attrition Distribution')
plt.tight_layout()
plt.show()

# Preprocessing
categorical_features = ['Education', 'City', 'Gender', 'EverBenched']
numerical_features = ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

X = df.drop('LeaveOrNot', axis=1)
y = df['LeaveOrNot']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) + numerical_features
importances = model.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Prediction example
sample_data = pd.DataFrame([{
    'Education': 'Masters',
    'JoiningYear': 2020,
    'City': 'Bangalore',
    'PaymentTier': 3,
    'Age': 30,
    'Gender': 'Female',
    'EverBenched': 'No',
    'ExperienceInCurrentDomain': 2
}])

sample_processed = preprocessor.transform(sample_data)
prediction = model.predict(sample_processed)
print(f"\nPrediction for sample data: {'Will Leave' if prediction[0] else 'Will Stay'}")