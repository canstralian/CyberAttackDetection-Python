"""
Data preprocessing module for cybersecurity detection.

This module provides functions for loading, cleaning, and preprocessing
cybersecurity datasets for machine learning model training and inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing cybersecurity datasets."""

    def __init__(self):
        """Initialize the preprocessor with default settings."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_fitted = False

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from a CSV file with error handling.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame or None if loading fails
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data with shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.error("File is empty")
            return None
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data and provide summary statistics.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }

        # Check for completely empty dataset
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")

        # Check for required columns (assuming 'label' is required)
        if 'label' not in data.columns:
            validation_results['warnings'].append(
                "No 'label' column found - assuming unsupervised learning"
            )

        # Check for excessive missing values
        missing_percentage = (data.isnull().sum() / len(data)) * 100
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        if high_missing_cols:
            validation_results['warnings'].append(
                f"Columns with >50% missing values: {high_missing_cols}"
            )

        return validation_results

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.

        Args:
            data: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()

        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_data[col].isnull().any():
                median_value = cleaned_data[col].median()
                cleaned_data.loc[:, col] = cleaned_data[col].fillna(median_value)
                logger.info(f"Filled missing values in {col} with median: {median_value}")

        # For categorical columns, fill with mode
        categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_data[col].isnull().any():
                mode_value = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'unknown'
                cleaned_data.loc[:, col] = cleaned_data[col].fillna(mode_value)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")

        return cleaned_data

    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with encoded categorical features
        """
        encoded_data = data.copy()

        # Get categorical columns (excluding the label column if present)
        categorical_cols = encoded_data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'label']

        if categorical_cols:
            # Use pandas get_dummies for one-hot encoding
            encoded_data = pd.get_dummies(
                encoded_data,
                columns=categorical_cols,
                prefix=categorical_cols,
                drop_first=True
            )
            logger.info(f"One-hot encoded columns: {categorical_cols}")

        return encoded_data

    def prepare_features_labels(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Separate features and labels from the dataset.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (features DataFrame, labels Series or None)
        """
        if 'label' in data.columns:
            X = data.drop('label', axis=1)
            y = data['label']
            logger.info(f"Separated features (shape: {X.shape}) and labels (unique: {y.nunique()})")
            return X, y
        else:
            logger.warning("No label column found - returning all columns as features")
            return data, None

    def scale_features(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Testing features (optional)

        Returns:
            Tuple of scaled training and testing features
        """
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.feature_columns = X_train.columns.tolist()
        self.is_fitted = True

        logger.info("Features scaled using StandardScaler")

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return X_train_scaled, None

    def full_preprocessing_pipeline(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
        """
        Complete preprocessing pipeline.

        Args:
            data: Input raw DataFrame

        Returns:
            Tuple of (processed features, labels, validation results)
        """
        # Validate data
        validation_results = self.validate_data(data)

        if not validation_results['is_valid']:
            logger.error("Data validation failed")
            return None, None, validation_results

        # Clean data
        cleaned_data = self.clean_data(data)

        # Encode categorical features
        encoded_data = self.encode_categorical_features(cleaned_data)

        # Separate features and labels
        X, y = self.prepare_features_labels(encoded_data)

        logger.info("Full preprocessing pipeline completed successfully")

        return X, y, validation_results


def load_and_preprocess(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Dict[str, Any]]:
    """
    Convenience function to load and preprocess data in one step.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (features, labels, validation results)
    """
    preprocessor = DataPreprocessor()

    # Load data
    data = preprocessor.load_data(file_path)
    if data is None:
        return None, None, {'is_valid': False, 'errors': ['Failed to load data']}

    # Run full preprocessing pipeline
    return preprocessor.full_preprocessing_pipeline(data)