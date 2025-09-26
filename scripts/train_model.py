#!/usr/bin/env python3
"""
Training script for cybersecurity attack detection models.

This script provides a command-line interface for training models
using the cybersecurity detection framework.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.preprocessing import DataPreprocessor
from src.models.detector import CyberAttackDetector, ModelRegistry
from src.utils.helpers import setup_logging, PerformanceTimer, ensure_directory
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train cybersecurity attack detection models'
    )
    
    # Data arguments
    parser.add_argument(
        '--data',
        required=True,
        help='Path to the training data CSV file'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        choices=list(ModelRegistry.get_available_models().keys()),
        default='random_forest',
        help='Type of model to train (default: random_forest)'
    )
    
    parser.add_argument(
        '--hyperparameter-tuning',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds for hyperparameter tuning (default: 5)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        default='models',
        help='Directory to save the trained model (default: models)'
    )
    
    parser.add_argument(
        '--model-name',
        help='Name for the saved model (default: auto-generated)'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (default: console only)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        # Validate input data file
        if not os.path.exists(args.data):
            logger.error(f"Data file not found: {args.data}")
            sys.exit(1)
        
        logger.info(f"Starting training with data: {args.data}")
        logger.info(f"Model type: {args.model}")
        logger.info(f"Test size: {args.test_size}")
        logger.info(f"Hyperparameter tuning: {args.hyperparameter_tuning}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        preprocessor = DataPreprocessor()
        
        with PerformanceTimer("Data preprocessing", logger):
            # Load data
            raw_data = pd.read_csv(args.data)
            logger.info(f"Loaded data with shape: {raw_data.shape}")
            
            # Preprocess data
            X, y, validation_results = preprocessor.full_preprocessing_pipeline(raw_data)
            
            if not validation_results['is_valid']:
                logger.error("Data validation failed:")
                for error in validation_results['errors']:
                    logger.error(f"  - {error}")
                sys.exit(1)
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    logger.warning(f"  - {warning}")
        
        # Split data
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        # Scale features
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Initialize model
        logger.info(f"Initializing {args.model} model...")
        detector = CyberAttackDetector(args.model)
        
        # Training
        if args.hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            with PerformanceTimer("Hyperparameter tuning", logger):
                tuning_results = detector.hyperparameter_tuning(
                    X_train_scaled, y_train, cv_folds=args.cv_folds
                )
            
            logger.info(f"Best parameters: {tuning_results['best_params']}")
            logger.info(f"Best CV score: {tuning_results['best_score']:.4f}")
            
        else:
            logger.info("Training model with default parameters...")
            with PerformanceTimer("Model training", logger):
                training_results = detector.train(
                    X_train_scaled, y_train, feature_names=X.columns.tolist()
                )
            
            logger.info(f"Training completed - CV accuracy: {training_results['cv_mean']:.4f}")
        
        # Evaluation
        logger.info("Evaluating model on test set...")
        with PerformanceTimer("Model evaluation", logger):
            evaluation_results = detector.evaluate(X_test_scaled, y_test)
        
        logger.info("Evaluation Results:")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_results['recall']:.4f}")
        logger.info(f"  F1 Score: {evaluation_results['f1_score']:.4f}")
        
        if 'auc_score' in evaluation_results:
            logger.info(f"  AUC Score: {evaluation_results['auc_score']:.4f}")
        
        # Save model
        ensure_directory(args.output_dir)
        
        if args.model_name:
            model_filename = f"{args.model_name}.pkl"
        else:
            model_filename = f"{args.model}_model.pkl"
        
        model_path = os.path.join(args.output_dir, model_filename)
        
        logger.info(f"Saving model to {model_path}...")
        detector.save_model(model_path)
        
        # Save training metadata
        metadata_path = os.path.join(args.output_dir, f"{Path(model_filename).stem}_metadata.json")
        
        metadata = {
            'model_type': args.model,
            'data_file': args.data,
            'data_shape': raw_data.shape,
            'test_size': args.test_size,
            'hyperparameter_tuning': args.hyperparameter_tuning,
            'validation_results': validation_results,
            'evaluation_results': evaluation_results,
            'model_path': model_path
        }
        
        if args.hyperparameter_tuning:
            metadata['tuning_results'] = tuning_results
        else:
            metadata['training_results'] = training_results
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to {metadata_path}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()