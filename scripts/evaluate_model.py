#!/usr/bin/env python3
"""
Evaluation script for cybersecurity attack detection models.

This script provides a command-line interface for evaluating
trained models on new datasets.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.preprocessing import DataPreprocessor
from src.models.detector import CyberAttackDetector
from src.utils.helpers import setup_logging, PerformanceTimer
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate cybersecurity attack detection models'
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        required=True,
        help='Path to the trained model file (.pkl)'
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='Path to the test data CSV file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save evaluation results (default: results)'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to CSV file'
    )
    
    parser.add_argument(
        '--plot-confusion-matrix',
        action='store_true',
        help='Generate and save confusion matrix plot'
    )
    
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
        # Validate input files
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            sys.exit(1)
        
        if not os.path.exists(args.data):
            logger.error(f"Data file not found: {args.data}")
            sys.exit(1)
        
        logger.info(f"Evaluating model: {args.model}")
        logger.info(f"Test data: {args.data}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load model
        logger.info("Loading model...")
        # Extract model type from filename (assuming format: modeltype_model.pkl)
        model_filename = Path(args.model).stem
        if '_model' in model_filename:
            model_type = model_filename.replace('_model', '')
        else:
            model_type = 'random_forest'  # Default fallback
        
        detector = CyberAttackDetector(model_type)
        detector.load_model(args.model)
        logger.info(f"Model loaded successfully (type: {model_type})")
        
        # Load and preprocess test data
        logger.info("Loading and preprocessing test data...")
        preprocessor = DataPreprocessor()
        
        with PerformanceTimer("Data preprocessing", logger):
            raw_data = pd.read_csv(args.data)
            logger.info(f"Loaded test data with shape: {raw_data.shape}")
            
            X, y, validation_results = preprocessor.full_preprocessing_pipeline(raw_data)
            
            if not validation_results['is_valid']:
                logger.error("Data validation failed:")
                for error in validation_results['errors']:
                    logger.error(f"  - {error}")
                sys.exit(1)
            
            # Scale features
            X_scaled, _ = preprocessor.scale_features(X)
        
        # Make predictions
        logger.info("Making predictions...")
        with PerformanceTimer("Prediction", logger):
            predictions = detector.predict(X_scaled)
            probabilities = detector.predict_proba(X_scaled)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        with PerformanceTimer("Model evaluation", logger):
            evaluation_results = detector.evaluate(X_scaled, y)
        
        # Display results
        logger.info("Evaluation Results:")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_results['recall']:.4f}")
        logger.info(f"  F1 Score: {evaluation_results['f1_score']:.4f}")
        
        if 'auc_score' in evaluation_results:
            logger.info(f"  AUC Score: {evaluation_results['auc_score']:.4f}")
        
        # Create detailed results
        results_summary = {
            'model_path': args.model,
            'model_type': model_type,
            'test_data_path': args.data,
            'test_data_shape': raw_data.shape,
            'validation_results': validation_results,
            'evaluation_results': evaluation_results,
            'predictions_count': len(predictions)
        }
        
        # Save results
        results_filename = f"evaluation_results_{Path(args.model).stem}_{Path(args.data).stem}.json"
        results_path = os.path.join(args.output_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions_df = pd.DataFrame({
                'true_label': y,
                'predicted_label': predictions
            })
            
            if probabilities is not None:
                # Add probability columns
                unique_classes = np.unique(y)
                for i, class_name in enumerate(unique_classes):
                    predictions_df[f'prob_{class_name}'] = probabilities[:, i]
            
            predictions_filename = f"predictions_{Path(args.model).stem}_{Path(args.data).stem}.csv"
            predictions_path = os.path.join(args.output_dir, predictions_filename)
            
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Predictions saved to {predictions_path}")
        
        # Generate confusion matrix plot if requested
        if args.plot_confusion_matrix:
            logger.info("Generating confusion matrix plot...")
            
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y, predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=np.unique(y), yticklabels=np.unique(y))
            plt.title(f'Confusion Matrix - {model_type.title()}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plot_filename = f"confusion_matrix_{Path(args.model).stem}_{Path(args.data).stem}.png"
            plot_path = os.path.join(args.output_dir, plot_filename)
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix plot saved to {plot_path}")
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(evaluation_results['classification_report'])
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()