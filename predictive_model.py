# predictive_model.py

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Libraries imported")

# Check current directory
print("Current directory:", os.getcwd())
print("Files present:", os.listdir())

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Dataset loaded successfully")
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Convert TotalCharges to numeric (some values are blank)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill NaN values without using inplace
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop irrelevant column
df = df.drop(['customerID'], axis=1)

# Encode target variable
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# One-hot encode only categorical columns
df = pd.get_dummies(df, columns=categorical_cols)

# Split Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain/Test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# FEATURE SELECTION: SelectKBest with chi-square
selector = SelectKBest(score_func=chi2, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("\nSelected Features:", selected_features.tolist())

# Train model on selected features
model = LogisticRegression(max_iter=2000)
model.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_selected)

print("\nModel Evaluation:")
print("Accuracy after SelectKBest:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (After SelectKBest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
