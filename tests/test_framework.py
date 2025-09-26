"""
Test suite for the cybersecurity detection framework.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.core.preprocessing import DataPreprocessor, load_and_preprocess
from src.models.detector import CyberAttackDetector, ModelRegistry, ModelComparer
from src.utils.helpers import (
    ensure_directory,
    validate_file_extension,
    sanitize_filename,
    create_sample_dataset,
    calculate_dataset_statistics
)


class TestDataPreprocessor:
    """Test cases for the DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        self.sample_data = create_sample_dataset(n_samples=100, n_features=5)
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        assert self.preprocessor.scaler is not None
        assert self.preprocessor.label_encoder is not None
        assert self.preprocessor.is_fitted is False
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        validation_results = self.preprocessor.validate_data(self.sample_data)
        
        assert validation_results['is_valid'] is True
        assert len(validation_results['errors']) == 0
        assert validation_results['shape'] == self.sample_data.shape
        assert 'label' in validation_results['columns']
    
    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        validation_results = self.preprocessor.validate_data(empty_data)
        
        assert validation_results['is_valid'] is False
        assert 'Dataset is empty' in validation_results['errors']
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:4, 'feature_01'] = np.nan
        data_with_missing.loc[0:2, 'protocol_type'] = np.nan
        
        cleaned_data = self.preprocessor.clean_data(data_with_missing)
        
        assert cleaned_data.isnull().sum().sum() == 0  # No missing values
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        encoded_data = self.preprocessor.encode_categorical_features(self.sample_data)
        
        # Check that categorical columns are encoded
        assert 'protocol_type' not in encoded_data.columns
        assert any('protocol_type_' in col for col in encoded_data.columns)
    
    def test_prepare_features_labels(self):
        """Test feature and label separation."""
        X, y = self.preprocessor.prepare_features_labels(self.sample_data)
        
        assert 'label' not in X.columns
        assert y is not None
        assert len(X) == len(y)
    
    def test_scale_features(self):
        """Test feature scaling."""
        X, y = self.preprocessor.prepare_features_labels(self.sample_data)
        X_numeric = X.select_dtypes(include=[np.number])
        
        X_scaled, _ = self.preprocessor.scale_features(X_numeric)
        
        assert self.preprocessor.is_fitted is True
        assert X_scaled.shape == X_numeric.shape
        # Check that scaled data has zero mean and unit variance (approximately)
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-7)
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-7)
    
    def test_full_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        X, y, validation_results = self.preprocessor.full_preprocessing_pipeline(self.sample_data)
        
        assert validation_results['is_valid'] is True
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert 'label' not in X.columns


class TestCyberAttackDetector:
    """Test cases for the CyberAttackDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CyberAttackDetector('random_forest')
        self.sample_data = create_sample_dataset(n_samples=100, n_features=10)
        
        # Prepare training data
        preprocessor = DataPreprocessor()
        X, y, _ = preprocessor.full_preprocessing_pipeline(self.sample_data)
        X_scaled, _ = preprocessor.scale_features(X.select_dtypes(include=[np.number]))
        
        self.X_train = X_scaled
        self.y_train = y
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.model_type == 'random_forest'
        assert self.detector.model is not None
        assert self.detector.is_trained is False
    
    def test_initialization_invalid_model(self):
        """Test detector initialization with invalid model type."""
        with pytest.raises(ValueError):
            CyberAttackDetector('invalid_model')
    
    def test_train(self):
        """Test model training."""
        training_results = self.detector.train(self.X_train, self.y_train)
        
        assert self.detector.is_trained is True
        assert 'training_time' in training_results
        assert 'cv_scores' in training_results
        assert training_results['training_samples'] == len(self.X_train)
    
    def test_predict_untrained(self):
        """Test prediction with untrained model."""
        with pytest.raises(ValueError):
            self.detector.predict(self.X_train)
    
    def test_predict_trained(self):
        """Test prediction with trained model."""
        self.detector.train(self.X_train, self.y_train)
        predictions = self.detector.predict(self.X_train)
        
        assert len(predictions) == len(self.X_train)
        assert all(pred in self.y_train.unique() for pred in predictions)
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.detector.train(self.X_train, self.y_train)
        evaluation_results = self.detector.evaluate(self.X_train, self.y_train)
        
        assert 'accuracy' in evaluation_results
        assert 'precision' in evaluation_results
        assert 'recall' in evaluation_results
        assert 'f1_score' in evaluation_results
        assert 0 <= evaluation_results['accuracy'] <= 1
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        self.detector.train(self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            self.detector.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new detector and load model
            new_detector = CyberAttackDetector('random_forest')
            new_detector.load_model(model_path)
            
            assert new_detector.is_trained is True
            assert new_detector.model_type == self.detector.model_type
            
            # Test predictions are consistent
            original_pred = self.detector.predict(self.X_train)
            loaded_pred = new_detector.predict(self.X_train)
            assert np.array_equal(original_pred, loaded_pred)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestModelRegistry:
    """Test cases for the ModelRegistry class."""
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = ModelRegistry.get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'random_forest' in models
        assert 'logistic_regression' in models
        
        for model_name, config in models.items():
            assert 'class' in config
            assert 'default_params' in config
            assert 'param_grid' in config


class TestModelComparer:
    """Test cases for the ModelComparer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparer = ModelComparer()
        self.sample_data = create_sample_dataset(n_samples=50, n_features=5)  # Small dataset for faster tests
        
        # Prepare data
        preprocessor = DataPreprocessor()
        X, y, _ = preprocessor.full_preprocessing_pipeline(self.sample_data)
        X_scaled, _ = preprocessor.scale_features(X.select_dtypes(include=[np.number]))
        
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
    
    def test_add_model(self):
        """Test adding models to comparer."""
        self.comparer.add_model('rf', 'random_forest')
        self.comparer.add_model('lr', 'logistic_regression')
        
        assert len(self.comparer.models) == 2
        assert 'rf' in self.comparer.models
        assert 'lr' in self.comparer.models
    
    def test_compare_models(self):
        """Test model comparison."""
        self.comparer.add_model('rf', 'random_forest')
        self.comparer.add_model('lr', 'logistic_regression')
        
        comparison_results = self.comparer.compare_models(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        assert 'detailed_results' in comparison_results
        assert 'summary' in comparison_results
        
        detailed_results = comparison_results['detailed_results']
        assert 'rf' in detailed_results
        assert 'lr' in detailed_results
        
        for model_name, results in detailed_results.items():
            assert 'training' in results
            assert 'evaluation' in results


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = os.path.join(tmp_dir, 'test', 'nested', 'directory')
            ensure_directory(test_path)
            assert os.path.exists(test_path)
    
    def test_validate_file_extension(self):
        """Test file extension validation."""
        allowed_extensions = {'csv', 'json', 'txt'}
        
        assert validate_file_extension('data.csv', allowed_extensions) is True
        assert validate_file_extension('config.json', allowed_extensions) is True
        assert validate_file_extension('readme.txt', allowed_extensions) is True
        assert validate_file_extension('image.png', allowed_extensions) is False
        assert validate_file_extension('data.CSV', allowed_extensions) is True  # Case insensitive
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        dangerous_filename = '../../../etc/passwd'
        sanitized = sanitize_filename(dangerous_filename)
        
        assert '/' not in sanitized
        assert '..' not in sanitized
        assert len(sanitized) > 0
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        dataset = create_sample_dataset(n_samples=100, n_features=5)
        
        assert len(dataset) == 100
        assert 'label' in dataset.columns
        assert 'protocol_type' in dataset.columns
        assert 'service' in dataset.columns
        
        # Check feature columns
        feature_cols = [col for col in dataset.columns if col.startswith('feature_')]
        assert len(feature_cols) == 5
    
    def test_calculate_dataset_statistics(self):
        """Test dataset statistics calculation."""
        dataset = create_sample_dataset(n_samples=50, n_features=5)
        stats = calculate_dataset_statistics(dataset)
        
        assert 'shape' in stats
        assert 'memory_usage_mb' in stats
        assert 'dtypes' in stats
        assert 'missing_values' in stats
        assert 'duplicate_rows' in stats
        
        assert stats['shape'] == dataset.shape
        assert stats['missing_values'] == 0  # Sample dataset has no missing values


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    dataset = create_sample_dataset(n_samples=50, n_features=3)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        dataset.to_csv(tmp_file.name, index=False)
        yield tmp_file.name
    
    os.unlink(tmp_file.name)


class TestIntegration:
    """Integration tests for the complete framework."""
    
    def test_end_to_end_pipeline(self, sample_csv_file):
        """Test complete end-to-end pipeline."""
        # Load and preprocess data
        X, y, validation_results = load_and_preprocess(sample_csv_file)
        
        assert validation_results['is_valid'] is True
        assert X is not None
        assert y is not None
        
        # Train model
        detector = CyberAttackDetector('random_forest')
        X_scaled, _ = DataPreprocessor().scale_features(X.select_dtypes(include=[np.number]))
        
        training_results = detector.train(X_scaled, y)
        assert detector.is_trained is True
        
        # Make predictions
        predictions = detector.predict(X_scaled)
        assert len(predictions) == len(X_scaled)
        
        # Evaluate model
        evaluation_results = detector.evaluate(X_scaled, y)
        assert 'accuracy' in evaluation_results