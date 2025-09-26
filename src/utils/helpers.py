"""
Utility functions for the cybersecurity detection framework.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {config_path}", e.doc, e.pos)


def save_json_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Dictionary with configuration values
        config_path: Path to save the configuration file
    """
    ensure_directory(os.path.dirname(config_path))
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def validate_file_extension(file_path: str, allowed_extensions: set) -> bool:
    """
    Validate if a file has an allowed extension.
    
    Args:
        file_path: Path to the file
        allowed_extensions: Set of allowed file extensions (without dots)
        
    Returns:
        True if the extension is allowed, False otherwise
    """
    file_extension = Path(file_path).suffix.lower().lstrip('.')
    return file_extension in allowed_extensions


def get_file_size_mb(file_path: str) -> float:
    """
    Get the size of a file in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in megabytes
    """
    return os.path.getsize(file_path) / (1024 * 1024)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove potentially dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and other potentially dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    sanitized = filename
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip().strip('.')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed_file'
    
    return sanitized


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def format_memory_usage(bytes_used: int) -> str:
    """
    Format memory usage in human-readable form.
    
    Args:
        bytes_used: Memory usage in bytes
        
    Returns:
        Formatted memory usage string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024.0:
            return f"{bytes_used:.1f} {unit}"
        bytes_used /= 1024.0
    return f"{bytes_used:.1f} TB"


def create_sample_dataset(n_samples: int = 1000, n_features: int = 20, 
                         n_classes: int = 2, noise: float = 0.1,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Create a sample cybersecurity dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes (attack types)
        noise: Amount of noise to add
        random_state: Random seed for reproducibility
        
    Returns:
        Generated DataFrame with features and labels
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=max(0, n_features // 4),
        n_informative=max(2, n_features // 2),  # Ensure at least 2 informative features
        random_state=random_state,
        flip_y=noise
    )
    
    # Create feature names
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features for realism
    df['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'], size=n_samples)
    df['service'] = np.random.choice(['http', 'ftp', 'ssh', 'smtp'], size=n_samples)
    
    # Add labels
    label_mapping = {0: 'normal', 1: 'attack'} if n_classes == 2 else {i: f'class_{i}' for i in range(n_classes)}
    df['label'] = [label_mapping[label] for label in y]
    
    return df


def calculate_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()
        stats['numeric_summary'] = numeric_stats.to_dict()
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        stats['categorical_summary'] = categorical_stats
    
    return stats


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the timer.
        
        Args:
            operation_name: Name of the operation being timed
            logger: Optional logger to use for output
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        import time
        elapsed_time = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {elapsed_time:.2f} seconds")
        else:
            self.logger.error(f"Failed {self.operation_name} after {elapsed_time:.2f} seconds")
        
        return False  # Don't suppress exceptions