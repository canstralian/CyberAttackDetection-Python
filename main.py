import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit app title
st.title("Cyber Attack Detection")

# Upload dataset
uploaded_file = st.file_uploader("Upload your cyber attack dataset (CSV file)", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    # Data Overview
    st.write("Dataset Shape:", data.shape)
    st.write("Columns:", data.columns)
    st.write("First few rows of the dataset:", data.head())
    
    # Label Distribution
    st.write("Label distribution:", data['label'].value_counts())
    
    # Preprocessing Steps
    data.fillna(0, inplace=True)
    data = pd.get_dummies(data, columns=['protocol_type'], drop_first=True)
    
    # Split data into features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model Performance Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    feature_importances = model.feature_importances_
    important_features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
    
    # Display metrics
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write("Confusion Matrix:", conf_matrix)
    st.write("Classification Report:", class_report)
    st.write("Top 10 important features:", important_features.head(10))
    
    # Save model performance metrics for further analysis
    performance_metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }

    # Allow user to download the performance metrics
    st.download_button(
        label="Download Metrics",
        data=pd.DataFrame(performance_metrics).to_csv().encode('utf-8'),
        file_name='performance_metrics.csv',
        mime='text/csv'
    )
else:
    st.write("Please upload a CSV file to proceed.")