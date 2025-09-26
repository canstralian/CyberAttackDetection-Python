"""
Flask API for cybersecurity attack detection framework.

This module provides a secure REST API for the cybersecurity detection system
with JWT authentication, rate limiting, and input validation.
"""

import os
import jwt
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError
import numpy as np
import pandas as pd

from ..core.preprocessing import DataPreprocessor
from ..models.detector import CyberAttackDetector, ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionRequestSchema(Schema):
    """Schema for detection API requests."""
    data = fields.List(fields.List(fields.Float()), required=True)
    model_type = fields.Str(missing='random_forest', validate=lambda x: x in ModelRegistry.get_available_models())


class TrainingRequestSchema(Schema):
    """Schema for model training API requests."""
    file_path = fields.Str(required=True)
    model_type = fields.Str(missing='random_forest', validate=lambda x: x in ModelRegistry.get_available_models())
    test_size = fields.Float(missing=0.2, validate=lambda x: 0.1 <= x <= 0.5)
    hyperparameter_tuning = fields.Bool(missing=False)


class APIConfig:
    """Configuration class for the API."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    JWT_EXPIRATION_DELTA = timedelta(hours=1)
    RATE_LIMIT_DEFAULT = "100/hour"
    RATE_LIMIT_AUTH = "20/minute"
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size


def create_app(config_class=APIConfig) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_class: Configuration class to use

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[app.config['RATE_LIMIT_DEFAULT']]
    )

    # Store limiter in app context for use in decorators
    app.limiter = limiter

    # Initialize global components
    app.preprocessor = DataPreprocessor()
    app.models = {}  # Cache for loaded models

    # Register error handlers
    register_error_handlers(app)

    # Register routes
    register_routes(app)

    # Security headers middleware
    @app.after_request
    def after_request(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

    logger.info("Flask application created and configured")
    return app


def register_error_handlers(app: Flask) -> None:
    """Register error handlers for the application."""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': 'Invalid request format or parameters'
        }), 400

    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'success': False,
            'error': 'Unauthorized',
            'message': 'Valid authentication token required'
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'success': False,
            'error': 'Forbidden',
            'message': 'Insufficient permissions'
        }), 403

    @app.errorhandler(429)
    def ratelimit_handler(error):
        return jsonify({
            'success': False,
            'error': 'Too Many Requests',
            'message': 'Rate limit exceeded. Please try again later.'
        }), 429

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500


def require_auth(f):
    """Decorator to require JWT authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({
                'success': False,
                'error': 'Missing token',
                'message': 'Authorization header is required'
            }), 401

        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]

            payload = jwt.decode(
                token,
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )

            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                return jsonify({
                    'success': False,
                    'error': 'Token expired',
                    'message': 'Please obtain a new token'
                }), 401

        except jwt.InvalidTokenError as e:
            return jsonify({
                'success': False,
                'error': 'Invalid token',
                'message': str(e)
            }), 401

        return f(*args, **kwargs)

    return decorated_function


def register_routes(app: Flask) -> None:
    """Register API routes."""

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'success': True,
            'message': 'Cybersecurity Detection API is running',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        })

    @app.route('/api/auth/token', methods=['POST'])
    @app.limiter.limit(app.config['RATE_LIMIT_AUTH'])
    def get_token():
        """Generate JWT token (simplified - in production, validate credentials)."""
        try:
            # In production, validate username/password here
            username = request.json.get('username', 'user')

            payload = {
                'username': username,
                'exp': datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA'],
                'iat': datetime.utcnow()
            }

            token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

            return jsonify({
                'success': True,
                'token': token,
                'expires_in': app.config['JWT_EXPIRATION_DELTA'].total_seconds()
            })

        except Exception as e:
            logger.error(f"Token generation error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Token generation failed',
                'message': str(e)
            }), 500

    @app.route('/api/models/available', methods=['GET'])
    def get_available_models():
        """Get list of available models."""
        try:
            available_models = ModelRegistry.get_available_models()
            model_info = {}

            for name, config in available_models.items():
                model_info[name] = {
                    'name': name,
                    'class': config['class'].__name__,
                    'default_params': config['default_params'],
                    'tunable_params': list(config['param_grid'].keys())
                }

            return jsonify({
                'success': True,
                'models': model_info
            })

        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to get available models',
                'message': str(e)
            }), 500

    @app.route('/api/detect', methods=['POST'])
    @require_auth
    def detect_attacks():
        """Detect cyber attacks using trained model."""
        try:
            # Validate input
            schema = DetectionRequestSchema()
            try:
                data = schema.load(request.json)
            except ValidationError as err:
                return jsonify({
                    'success': False,
                    'error': 'Validation error',
                    'message': err.messages
                }), 400

            # Get or load model
            model_type = data['model_type']
            model_key = f"detector_{model_type}"

            if model_key not in app.models:
                # Try to load pre-trained model
                model_path = f"models/{model_type}_model.pkl"
                if os.path.exists(model_path):
                    detector = CyberAttackDetector(model_type)
                    detector.load_model(model_path)
                    app.models[model_key] = detector
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Model not found',
                        'message': f'No trained model available for {model_type}'
                    }), 404

            # Make predictions
            detector = app.models[model_key]
            input_data = np.array(data['data'])

            predictions = detector.predict(input_data)
            probabilities = detector.predict_proba(input_data)

            response_data = {
                'success': True,
                'predictions': predictions.tolist(),
                'model_type': model_type,
                'num_samples': len(predictions)
            }

            if probabilities is not None:
                response_data['probabilities'] = probabilities.tolist()

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Detection failed',
                'message': str(e)
            }), 500

    @app.route('/api/train', methods=['POST'])
    @require_auth
    def train_model():
        """Train a new model."""
        try:
            # Validate input
            schema = TrainingRequestSchema()
            try:
                data = schema.load(request.json)
            except ValidationError as err:
                return jsonify({
                    'success': False,
                    'error': 'Validation error',
                    'message': err.messages
                }), 400

            # Load and preprocess data
            file_path = data['file_path']
            if not os.path.exists(file_path):
                return jsonify({
                    'success': False,
                    'error': 'File not found',
                    'message': f'Data file not found: {file_path}'
                }), 404

            X, y, validation_results = app.preprocessor.full_preprocessing_pipeline(
                pd.read_csv(file_path)
            )

            if not validation_results['is_valid']:
                return jsonify({
                    'success': False,
                    'error': 'Data validation failed',
                    'message': validation_results['errors']
                }), 400

            # Split data
            from sklearn.model_selection import train_test_split
            test_size = data['test_size']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scale features
            X_train_scaled, X_test_scaled = app.preprocessor.scale_features(X_train, X_test)

            # Initialize and train model
            model_type = data['model_type']
            detector = CyberAttackDetector(model_type)

            # Perform hyperparameter tuning if requested
            if data['hyperparameter_tuning']:
                tuning_results = detector.hyperparameter_tuning(X_train_scaled, y_train)
            else:
                training_results = detector.train(X_train_scaled, y_train, X.columns.tolist())

            # Evaluate model
            evaluation_results = detector.evaluate(X_test_scaled, y_test)

            # Save model
            model_path = f"models/{model_type}_model.pkl"
            os.makedirs('models', exist_ok=True)
            detector.save_model(model_path)

            # Cache model
            app.models[f"detector_{model_type}"] = detector

            response_data = {
                'success': True,
                'model_type': model_type,
                'model_path': model_path,
                'evaluation': evaluation_results,
                'validation': validation_results
            }

            if data['hyperparameter_tuning']:
                response_data['hyperparameter_tuning'] = tuning_results
            else:
                response_data['training'] = training_results

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Training failed',
                'message': str(e)
            }), 500

    @app.route('/api/models/<model_type>/evaluate', methods=['POST'])
    @require_auth
    def evaluate_model(model_type: str):
        """Evaluate a trained model on test data."""
        try:
            if model_type not in ModelRegistry.get_available_models():
                return jsonify({
                    'success': False,
                    'error': 'Invalid model type',
                    'message': f'Model type {model_type} not supported'
                }), 400

            # Get test data file path from request
            file_path = request.json.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return jsonify({
                    'success': False,
                    'error': 'File not found',
                    'message': 'Test data file not found'
                }), 404

            # Load model
            model_key = f"detector_{model_type}"
            if model_key not in app.models:
                model_path = f"models/{model_type}_model.pkl"
                if not os.path.exists(model_path):
                    return jsonify({
                        'success': False,
                        'error': 'Model not found',
                        'message': f'No trained model available for {model_type}'
                    }), 404

                detector = CyberAttackDetector(model_type)
                detector.load_model(model_path)
                app.models[model_key] = detector

            # Preprocess test data
            X, y, _ = app.preprocessor.full_preprocessing_pipeline(pd.read_csv(file_path))
            X_scaled, _ = app.preprocessor.scale_features(X)

            # Evaluate model
            detector = app.models[model_key]
            evaluation_results = detector.evaluate(X_scaled, y)

            return jsonify({
                'success': True,
                'model_type': model_type,
                'evaluation': evaluation_results
            })

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Evaluation failed',
                'message': str(e)
            }), 500


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=5000)