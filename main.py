# Cyber Attack Detection Python Script

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- ### Training Data Details ---
# Loading a sample dataset (replace this with your actual cyber attack dataset)
# For example, using the UNSW-NB15 dataset (replace the path with your dataset path)
data = pd.read_csv('path_to_dataset.csv')

# Data Overview: Shape, Columns, and Initial Records
print(f"Dataset Shape: {data.shape}")
print(f"Columns: {data.columns}")
print("First few rows of the dataset:\n", data.head())

# Label Distribution (Assuming 'label' is the target variable)
print("Label distribution:\n", data['label'].value_counts())

# --- ### Preprocessing Steps ---

# 1. Handle missing values (if any)
data.fillna(0, inplace=True)

# 2. Encode categorical features (if any)
# Assuming 'protocol_type' is a categorical feature
data = pd.get_dummies(data, columns=['protocol_type'], drop_first=True)

# 3. Split the data into features (X) and target (y)
X = data.drop('label', axis=1)
y = data['label']

# 4. Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- ### Model Architecture ---

# We will use a RandomForestClassifier for cyber attack detection
model = RandomForestClassifier(n_estimators=100, random_state=42)

# --- ### Hyperparameters ---

# Hyperparameters for RandomForestClassifier:
# - n_estimators: The number of trees in the forest (100)
# - criterion: Function to measure the quality of the split ('gini')
# - max_depth: Maximum depth of the tree (None, default)
# - random_state: Ensures reproducibility (42)

# Train the model
model.fit(X_train, y_train)

# --- ### Model Performance Metrics ---

# 1. Predict on the test data
y_pred = model.predict(X_test)

# 2. Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 3. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 4. Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# 5. Feature Importance (Top 10 most important features)
feature_importances = model.feature_importances_
important_features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
print("Top 10 important features:\n", important_features.head(10))

# Save model performance metrics for further analysis
performance_metrics = {
    'accuracy': accuracy,
    'confusion_matrix': conf_matrix.tolist(),
    'classification_report': class_report
}