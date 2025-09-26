"""
Configuration module for cybersecurity detection framework.
"""

import os
from datetime import timedelta
from typing import List


class Config:
    """Base configuration class."""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # API settings
    JWT_EXPIRATION_DELTA = timedelta(hours=int(os.environ.get('JWT_EXPIRATION_HOURS', 1)))
    RATE_LIMIT_DEFAULT = f"{os.environ.get('RATE_LIMIT_PER_HOUR', 100)}/hour"
    RATE_LIMIT_AUTH = f"{os.environ.get('RATE_LIMIT_AUTH_PER_MINUTE', 20)}/minute"
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE_MB', 16)) * 1024 * 1024
    
    # CORS settings
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # File paths
    MODEL_DIRECTORY = os.environ.get('MODEL_DIRECTORY', 'models/')
    DATA_DIRECTORY = os.environ.get('DATA_DIRECTORY', 'data/')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/cyberattack_detection.log')
    
    # Model settings
    DEFAULT_MODEL_TYPE = os.environ.get('DEFAULT_MODEL_TYPE', 'random_forest')
    MODEL_CACHE_SIZE = int(os.environ.get('MODEL_CACHE_SIZE', 5))
    
    # Security settings
    ALLOWED_EXTENSIONS = {'csv', 'json'}
    
    @staticmethod
    def init_app(app):
        """Initialize application with config."""
        pass


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    LOG_LEVEL = 'INFO'
    
    # Override with stronger security settings
    CORS_ORIGINS = []  # Configure based on production needs
    
    @classmethod
    def init_app(cls, app):
        """Initialize production app."""
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.INFO)
        app.logger.addHandler(syslog_handler)


class TestingConfig(Config):
    """Testing configuration."""
    
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Use in-memory database for testing
    DATABASE_URL = 'sqlite:///:memory:'
    
    # Disable rate limiting for tests
    RATE_LIMIT_DEFAULT = "1000/hour"
    RATE_LIMIT_AUTH = "100/minute"


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}