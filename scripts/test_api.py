#!/usr/bin/env python3
"""
Simple API client for testing the cybersecurity detection API.
"""

import requests
import json
import sys
import numpy as np

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:5000"
    
    print("Testing Cybersecurity Detection API...")
    
    # 1. Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False
    
    # 2. Get authentication token
    print("\n2. Getting authentication token...")
    try:
        auth_data = {"username": "test_user"}
        response = requests.post(f"{base_url}/api/auth/token", json=auth_data, timeout=5)
        if response.status_code == 200:
            token = response.json()['token']
            print("✓ Token obtained successfully")
        else:
            print(f"✗ Token request failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Token request error: {e}")
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Test available models
    print("\n3. Testing available models endpoint...")
    try:
        response = requests.get(f"{base_url}/api/models/available", timeout=5)
        if response.status_code == 200:
            models = response.json()['models']
            print("✓ Available models retrieved")
            print(f"  Available models: {list(models.keys())}")
        else:
            print(f"✗ Available models failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Available models error: {e}")
    
    # 4. Test detection (with sample data)
    print("\n4. Testing detection endpoint...")
    try:
        # Generate some sample data for detection
        sample_data = np.random.randn(5, 15).tolist()  # 5 samples, 15 features
        
        detection_data = {
            "data": sample_data,
            "model_type": "random_forest"
        }
        
        response = requests.post(
            f"{base_url}/api/detect",
            json=detection_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Detection successful")
            print(f"  Predictions: {result.get('predictions', [])}")
            print(f"  Model used: {result.get('model_type')}")
        elif response.status_code == 404:
            print("! Model not found - this is expected if no model is trained yet")
            print("  Train a model first using: python scripts/train_model.py")
        else:
            print(f"✗ Detection failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Detection error: {e}")
    
    print("\n5. API test completed!")
    return True

if __name__ == '__main__':
    success = test_api()
    sys.exit(0 if success else 1)