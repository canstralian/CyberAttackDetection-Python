#!/usr/bin/env python3
"""
Data generation script for cybersecurity attack detection.

This script generates sample cybersecurity datasets for training and testing.
"""

import argparse
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.helpers import create_sample_dataset, setup_logging, ensure_directory
import pandas as pd


def main():
    """Main data generation function."""
    parser = argparse.ArgumentParser(
        description='Generate sample cybersecurity datasets'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--features',
        type=int,
        default=20,
        help='Number of features to generate (default: 20)'
    )
    
    parser.add_argument(
        '--classes',
        type=int,
        default=2,
        help='Number of classes (attack types) (default: 2)'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=0.1,
        help='Amount of noise to add (0.0-1.0) (default: 0.1)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir',
        default='data',
        help='Directory to save the generated data (default: data)'
    )
    
    parser.add_argument(
        '--filename',
        help='Output filename (default: auto-generated)'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Generate separate train/test files'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion when splitting (default: 0.2)'
    )
    
    # Logging parameters
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("Generating sample cybersecurity dataset...")
        logger.info(f"Samples: {args.samples}")
        logger.info(f"Features: {args.features}")
        logger.info(f"Classes: {args.classes}")
        logger.info(f"Noise level: {args.noise}")
        
        # Generate dataset
        dataset = create_sample_dataset(
            n_samples=args.samples,
            n_features=args.features,
            n_classes=args.classes,
            noise=args.noise,
            random_state=args.random_state
        )
        
        logger.info(f"Generated dataset with shape: {dataset.shape}")
        
        # Create output directory
        ensure_directory(args.output_dir)
        
        # Determine filename
        if args.filename:
            base_filename = args.filename
        else:
            base_filename = f"cyber_dataset_{args.samples}s_{args.features}f_{args.classes}c"
        
        if args.split:
            # Split into train and test sets
            from sklearn.model_selection import train_test_split
            
            X = dataset.drop('label', axis=1)
            y = dataset['label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
            )
            
            train_dataset = pd.concat([X_train, y_train], axis=1)
            test_dataset = pd.concat([X_test, y_test], axis=1)
            
            # Save train set
            train_filename = f"{base_filename}_train.csv"
            train_path = os.path.join(args.output_dir, train_filename)
            train_dataset.to_csv(train_path, index=False)
            logger.info(f"Training data saved to {train_path} (shape: {train_dataset.shape})")
            
            # Save test set
            test_filename = f"{base_filename}_test.csv"
            test_path = os.path.join(args.output_dir, test_filename)
            test_dataset.to_csv(test_path, index=False)
            logger.info(f"Test data saved to {test_path} (shape: {test_dataset.shape})")
            
        else:
            # Save complete dataset
            if not base_filename.endswith('.csv'):
                base_filename += '.csv'
            
            output_path = os.path.join(args.output_dir, base_filename)
            dataset.to_csv(output_path, index=False)
            logger.info(f"Dataset saved to {output_path}")
        
        # Display dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"  Total samples: {len(dataset)}")
        logger.info(f"  Features: {len(dataset.columns) - 1}")  # Exclude label column
        logger.info(f"  Memory usage: {dataset.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        logger.info("\nLabel Distribution:")
        label_counts = dataset['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(dataset)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        logger.info("\nFeature Statistics:")
        numeric_features = dataset.select_dtypes(include=['float64', 'int64'])
        logger.info(f"  Numeric features: {len(numeric_features.columns)}")
        
        categorical_features = dataset.select_dtypes(include=['object'])
        categorical_features = categorical_features.drop('label', axis=1, errors='ignore')
        logger.info(f"  Categorical features: {len(categorical_features.columns)}")
        
        logger.info("Data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data generation failed with error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()