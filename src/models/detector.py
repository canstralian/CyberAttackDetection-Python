"""
Machine learning models for cybersecurity attack detection.

This module provides classes and functions for training, evaluating,
and managing machine learning models for cybersecurity threat detection.
"""

import pickle
import json
import os
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for available machine learning models."""

    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """
        Get dictionary of available models with their configurations.

        Returns:
            Dictionary mapping model names to their configurations
        """
        return {
            'random_forest': {
                'class': RandomForestClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 10,
                    'min_samples_split': 5
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'default_params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'C': 1.0
                },
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'class': SVC,
                'default_params': {
                    'random_state': 42,
                    'probability': True,
                    'C': 1.0,
                    'kernel': 'rbf'
                },
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'neural_network': {
                'class': MLPClassifier,
                'default_params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'hidden_layer_sizes': (100,),
                    'alpha': 0.0001
                },
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }


class CyberAttackDetector:
    """Main class for cybersecurity attack detection models."""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the detector with a specific model type.

        Args:
            model_type: Type of model to use ('random_forest', 'logistic_regression', 'svm', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.model_config = None
        self.training_history = {}
        self.feature_names = None
        self.is_trained = False

        # Get model configuration
        available_models = ModelRegistry.get_available_models()
        if model_type not in available_models:
            raise ValueError(f"Model type '{model_type}' not supported. "
                           f"Available models: {list(available_models.keys())}")

        self.model_config = available_models[model_type]
        self.model = self.model_config['class'](**self.model_config['default_params'])

        logger.info(f"Initialized {model_type} detector")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the model on the provided dataset.

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features (optional)

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {self.model_type} model...")

        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        # Train the model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')

        # Store training history
        self.training_history = {
            'model_type': self.model_type,
            'training_time': training_time,
            'training_samples': X_train.shape[0],
            'features_count': X_train.shape[1],
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'timestamp': datetime.now().isoformat()
        }

        self.is_trained = True

        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features to predict

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities if supported by the model.

        Args:
            X: Features to predict

        Returns:
            Prediction probabilities or None if not supported
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            logger.warning(f"Model {self.model_type} does not support probability predictions")
            return None

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model performance on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }

        # Add AUC score if probabilities are available
        if y_proba is not None and len(np.unique(y_test)) == 2:
            metrics['auc_score'] = roc_auc_score(y_test, y_proba[:, 1])

        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            metrics['feature_importance'] = feature_importance.to_dict()

        logger.info(f"Model evaluation completed - Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with tuning results
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type}...")

        # Get parameter grid
        param_grid = self.model_config['param_grid']

        # Initialize fresh model
        base_model = self.model_config['class']()

        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        start_time = datetime.now()
        grid_search.fit(X_train, y_train)
        tuning_time = (datetime.now() - start_time).total_seconds()

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True

        # Store tuning results
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'tuning_time': tuning_time,
            'n_combinations': len(grid_search.cv_results_['params']),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return tuning_results

    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            file_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            file_path: Path to the saved model
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data['training_history']
            self.is_trained = True

            logger.info(f"Model loaded from {file_path}")

        except FileNotFoundError:
            logger.error(f"Model file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


class ModelComparer:
    """Class for comparing multiple models."""

    def __init__(self):
        """Initialize the model comparer."""
        self.models = {}
        self.comparison_results = {}

    def add_model(self, name: str, model_type: str) -> None:
        """
        Add a model to the comparison.

        Args:
            name: Unique name for the model
            model_type: Type of model ('random_forest', 'logistic_regression', etc.)
        """
        self.models[name] = CyberAttackDetector(model_type)

    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare all added models on the same dataset.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(self.models)} models...")

        self.comparison_results = {}

        for name, model in self.models.items():
            logger.info(f"Training and evaluating {name}...")

            # Train model
            training_results = model.train(X_train, y_train, feature_names)

            # Evaluate model
            evaluation_results = model.evaluate(X_test, y_test)

            # Store results
            self.comparison_results[name] = {
                'training': training_results,
                'evaluation': evaluation_results
            }

        # Create comparison summary
        summary = self._create_comparison_summary()

        logger.info("Model comparison completed")

        return {
            'detailed_results': self.comparison_results,
            'summary': summary
        }

    def _create_comparison_summary(self) -> Dict[str, Any]:
        """Create a summary of the model comparison."""
        if not self.comparison_results:
            return {}

        summary = {
            'best_accuracy': {'model': '', 'score': 0},
            'best_precision': {'model': '', 'score': 0},
            'best_recall': {'model': '', 'score': 0},
            'best_f1': {'model': '', 'score': 0},
            'fastest_training': {'model': '', 'time': float('inf')},
            'accuracy_ranking': []
        }

        for name, results in self.comparison_results.items():
            eval_results = results['evaluation']
            train_results = results['training']

            # Check best accuracy
            if eval_results['accuracy'] > summary['best_accuracy']['score']:
                summary['best_accuracy'] = {'model': name, 'score': eval_results['accuracy']}

            # Check best precision
            if eval_results['precision'] > summary['best_precision']['score']:
                summary['best_precision'] = {'model': name, 'score': eval_results['precision']}

            # Check best recall
            if eval_results['recall'] > summary['best_recall']['score']:
                summary['best_recall'] = {'model': name, 'score': eval_results['recall']}

            # Check best F1 score
            if eval_results['f1_score'] > summary['best_f1']['score']:
                summary['best_f1'] = {'model': name, 'score': eval_results['f1_score']}

            # Check fastest training
            if train_results['training_time'] < summary['fastest_training']['time']:
                summary['fastest_training'] = {'model': name, 'time': train_results['training_time']}

        # Create accuracy ranking
        summary['accuracy_ranking'] = sorted(
            [(name, results['evaluation']['accuracy']) for name, results in self.comparison_results.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return summary