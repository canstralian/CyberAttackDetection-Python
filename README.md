# Cybersecurity Attack Detection Framework

![🔧 Build Status](https://github.com/canstralian/CyberAttackDetection-Python/actions/workflows/ci.yml/badge.svg)  
![📊 Coverage](https://codecov.io/gh/canstralian/CyberAttackDetection-Python/branch/main/graph/badge.svg)  
![📦 Dependencies](https://img.shields.io/librariesio/release/github/canstralian/CyberAttackDetection-Python)  
![📜 License](https://img.shields.io/github/license/canstralian/CyberAttackDetection-Python)  
![🕒 Last Commit](https://img.shields.io/github/last-commit/canstralian/CyberAttackDetection-Python)  
![🚀 Release](https://img.shields.io/github/v/release/canstralian/CyberAttackDetection-Python)  
![🐞 Issues](https://img.shields.io/github/issues/canstralian/CyberAttackDetection-Python)

## Overview

A modular, secure, and extensible framework for detecting cyber attacks using machine learning. This framework provides a robust foundation for cybersecurity threat detection with built-in best practices for data preprocessing, model management, REST API development, and security.

### Key Features

- **🛡️ Security-First Design**: JWT authentication, rate limiting, input validation, and security headers
- **🔧 Modular Architecture**: Clean separation of concerns with dedicated modules for preprocessing, modeling, and API
- **📊 Multiple ML Models**: Support for Random Forest, Logistic Regression, SVM, and Neural Networks
- **🚀 REST API**: Secure Flask-based API with comprehensive validation using Marshmallow
- **📈 Comprehensive Evaluation**: Built-in metrics, cross-validation, and hyperparameter tuning
- **🧪 Testing Framework**: Extensive test suite with pytest and continuous integration
- **📖 Documentation**: Jupyter notebooks, scripts, and comprehensive API documentation
- **🔄 Model Management**: Easy model saving, loading, and comparison capabilities

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/canstralian/CyberAttackDetection-Python.git
cd CyberAttackDetection-Python

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Generate Sample Data
```bash
python scripts/generate_data.py --samples 1000 --features 20 --output-dir data/
```

#### 2. Train a Model
```bash
python scripts/train_model.py --data data/cyber_dataset_1000s_20f_2c.csv --model random_forest --hyperparameter-tuning
```

#### 3. Start the API Server
```bash
cd src/api && python app.py
```

#### 4. Use the Framework Programmatically

```python
from src.core.preprocessing import DataPreprocessor
from src.models.detector import CyberAttackDetector
from src.utils.helpers import create_sample_dataset

# Generate sample data
data = create_sample_dataset(n_samples=1000, n_features=20)

# Preprocess data
preprocessor = DataPreprocessor()
X, y, _ = preprocessor.full_preprocessing_pipeline(data)
X_scaled, _ = preprocessor.scale_features(X.select_dtypes(include=['float64', 'int64']))

# Train model
detector = CyberAttackDetector('random_forest')
detector.train(X_scaled, y)

# Make predictions
predictions = detector.predict(X_scaled)
evaluation = detector.evaluate(X_scaled, y)
print(f"Accuracy: {evaluation['accuracy']:.4f}")
```

## Directory Structure

```
CyberAttackDetection-Python/
├── .github/                    # GitHub workflows and templates
│   └── workflows/
│       ├── ci.yml             # Continuous integration
│       ├── security-scan.yml  # Security scanning
│       └── python-app.yml     # Python application workflow
├── src/                       # Main source code
│   ├── __init__.py
│   ├── api/                   # REST API module
│   │   ├── __init__.py
│   │   └── app.py            # Flask application with security features
│   ├── core/                  # Core detection and preprocessing
│   │   ├── __init__.py
│   │   └── preprocessing.py   # Data preprocessing pipeline
│   ├── models/                # Machine learning models
│   │   ├── __init__.py
│   │   └── detector.py       # Model classes and management
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── helpers.py        # Helper functions and utilities
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_framework.py     # Comprehensive tests
├── notebooks/                 # Jupyter notebooks
│   └── model_training_demo.ipynb  # Training demonstration
├── scripts/                   # Command-line scripts
│   ├── train_model.py        # Model training script
│   ├── evaluate_model.py     # Model evaluation script
│   └── generate_data.py      # Data generation script
├── config/                    # Configuration files
│   ├── .env.example          # Environment variables template
│   └── config.py             # Application configuration
├── data/                      # Data directory
├── models/                    # Saved models directory
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## API Usage

### Authentication

First, obtain a JWT token:

```bash
curl -X POST http://localhost:5000/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user"}'
```

### Available Endpoints

- `GET /api/health` - Health check
- `POST /api/auth/token` - Get authentication token
- `GET /api/models/available` - List available models
- `POST /api/detect` - Detect attacks (requires auth)
- `POST /api/train` - Train new model (requires auth)
- `POST /api/models/{model_type}/evaluate` - Evaluate model (requires auth)

### Detection Example

```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "data": [[1.2, 0.5, -0.3, ...], [0.8, -1.1, 0.4, ...]],
    "model_type": "random_forest"
  }'
```

## Model Types

The framework supports multiple machine learning algorithms:

- **Random Forest** (`random_forest`): Ensemble method with excellent performance
- **Logistic Regression** (`logistic_regression`): Fast linear classifier
- **Support Vector Machine** (`svm`): Powerful non-linear classifier
- **Neural Network** (`neural_network`): Multi-layer perceptron

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Protection against API abuse
- **Input Validation**: Comprehensive request validation using Marshmallow
- **Security Headers**: Standard security headers (HSTS, XSS Protection, etc.)
- **CORS Protection**: Configurable cross-origin resource sharing
- **Error Handling**: Secure error responses without information leakage

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_framework.py
```

### Code Quality

```bash
# Linting
flake8 src/ tests/ scripts/

# Code formatting
black src/ tests/ scripts/

# Type checking (if using mypy)
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following PEP 8 guidelines
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Configuration

### Environment Variables

Copy `config/.env.example` to `.env` and customize:

```bash
# Security
SECRET_KEY=your-secret-key-here
JWT_EXPIRATION_HOURS=1

# API Configuration
RATE_LIMIT_PER_HOUR=100
RATE_LIMIT_AUTH_PER_MINUTE=20

# Model Configuration
DEFAULT_MODEL_TYPE=random_forest
MODEL_DIRECTORY=models/

# Data Configuration
DATA_DIRECTORY=data/
MAX_FILE_SIZE_MB=16
```

## Jupyter Notebooks

Explore the framework with interactive notebooks:

- `notebooks/model_training_demo.ipynb`: Complete training and evaluation workflow
- Examples of data preprocessing, model comparison, and hyperparameter tuning

## Performance and Scalability

- **Efficient preprocessing**: Optimized data pipeline with memory management
- **Model caching**: Intelligent model loading and caching in the API
- **Batch processing**: Support for batch predictions
- **Async operations**: Performance timers and logging for optimization

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the project root directory
2. **Model Not Found**: Check if models are saved in the correct directory
3. **Authentication Errors**: Verify JWT token is valid and not expired
4. **Memory Issues**: Reduce batch size or dataset size for large datasets

### Logging

The framework includes comprehensive logging. Set log levels:

```python
from src.utils.helpers import setup_logging
logger = setup_logging('DEBUG', 'logs/debug.log')
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Flask](https://flask.palletsprojects.com/) for the REST API framework
- [Marshmallow](https://marshmallow.readthedocs.io/) for input validation
- [PyJWT](https://pyjwt.readthedocs.io/) for JWT authentication
- [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) for data processing

---

Built with ❤️ for cybersecurity professionals and researchers.
