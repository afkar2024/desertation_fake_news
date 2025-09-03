#!/usr/bin/env python3
"""
Test API Integration with Enhanced Model Service
Verifies that all endpoints work correctly with the new model service
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_basic_prediction():
    """Test basic prediction endpoint"""
    print("🧪 Testing Basic Prediction Endpoint")
    print("-" * 40)
    
    test_text = "Bill Gates will give you $5000 if you share this Facebook post!"
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": test_text}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prediction successful")
        print(f"   Text: {test_text[:50]}...")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Confidence: {data['confidence']:.2%}")
        print(f"   Probabilities: {data['explanation']['probabilities']}")
        return True
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_prediction_with_explanation():
    """Test prediction with explanation endpoint"""
    print("\n🧪 Testing Prediction with Explanation")
    print("-" * 40)
    
    test_text = "Scientists discover miracle cure that doctors hate!"
    
    response = requests.post(
        f"{BASE_URL}/predict/explain",
        json={"text": test_text, "top_tokens": 10}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prediction with explanation successful")
        print(f"   Text: {test_text[:50]}...")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Confidence: {data['confidence']:.2%}")
        print(f"   Model Used: {data.get('model_used', 'unknown')}")
        print(f"   Expected Performance: {data.get('expected_performance', {})}")
        print(f"   Explanation Tokens: {data['explanation'].get('tokens', [])[:5]}")
        return True
    else:
        print(f"❌ Prediction with explanation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🧪 Testing Model Info Endpoint")
    print("-" * 40)
    
    response = requests.get(f"{BASE_URL}/model/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model info retrieved successfully")
        print(f"   Strategy: {data['strategy']}")
        print(f"   Device: {data['device']}")
        print(f"   Loaded Models: {len(data['loaded_models'])}")
        for model in data['loaded_models']:
            print(f"     - {model.get('name', model.get('key', 'unknown'))}")
        return True
    else:
        print(f"❌ Model info failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_shap_explanation():
    """Test SHAP explanation endpoint"""
    print("\n🧪 Testing SHAP Explanation Endpoint")
    print("-" * 40)
    
    test_text = "You won't believe what happened next!"
    
    response = requests.post(
        f"{BASE_URL}/explain/shap",
        json={"text": test_text}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SHAP explanation successful")
        print(f"   Text: {test_text[:50]}...")
        print(f"   Tokens: {data['tokens'][:5]}")
        print(f"   SHAP Values: {data['shap_values'][:5]}")
        print(f"   Base Value: {data['base_value']}")
        print(f"   Message: {data.get('message', 'No message')}")
        return True
    else:
        print(f"❌ SHAP explanation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_counterfactuals():
    """Test counterfactuals endpoint"""
    print("\n🧪 Testing Counterfactuals Endpoint")
    print("-" * 40)
    
    test_text = "This is a fake news article"
    
    response = requests.post(
        f"{BASE_URL}/predict/counterfactual",
        json={"text": test_text, "max_candidates": 2}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Counterfactuals generated successfully")
        print(f"   Base Text: {data['base_text']}")
        print(f"   Counterfactuals: {len(data['counterfactuals'])}")
        for i, cf in enumerate(data['counterfactuals']):
            print(f"     {i+1}. Removed: '{cf.get('removed_word', 'unknown')}'")
            print(f"        Text: {cf['text'][:50]}...")
            print(f"        Prediction: {'FAKE' if cf['prediction']['prediction'] == 1 else 'REAL'}")
        return True
    else:
        print(f"❌ Counterfactuals failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🧪 Testing Batch Prediction Endpoint")
    print("-" * 40)
    
    test_texts = [
        "Bill Gates scam post",
        "Federal Reserve announces interest rate decision",
        "Aliens land in New York City"
    ]
    
    requests_data = [{"text": text} for text in test_texts]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=requests_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Batch prediction successful")
        print(f"   Total Processed: {data['total_processed']}")
        for i, pred in enumerate(data['predictions']):
            print(f"   {i+1}. Text: {test_texts[i][:30]}...")
            print(f"      Prediction: {pred['prediction']}")
            print(f"      Confidence: {pred['confidence']:.2%}")
        return True
    else:
        print(f"❌ Batch prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_health_check():
    """Test health check endpoint"""
    print("\n🧪 Testing Health Check Endpoint")
    print("-" * 40)
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check successful")
        print(f"   Status: {data['status']}")
        print(f"   Articles Count: {data['articles_count']}")
        print(f"   Sources Count: {data['sources_count']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Testing API Integration with Enhanced Model Service")
    print("=" * 60)
    print("This script tests all major endpoints to ensure they work")
    print("with the enhanced model service (Pulk17/Fake-News-Detection)")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("\n⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test all endpoints
    tests = [
        ("Health Check", test_health_check),
        ("Basic Prediction", test_basic_prediction),
        ("Prediction with Explanation", test_prediction_with_explanation),
        ("Model Info", test_model_info),
        ("SHAP Explanation", test_shap_explanation),
        ("Counterfactuals", test_counterfactuals),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total:.1%}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Enhanced model service is fully integrated")
        print("✅ All analytics endpoints are working")
        print("✅ API is ready for frontend consumption")
        
        print("\n📋 Next Steps:")
        print("1. Start the frontend application")
        print("2. Test the complete user workflow")
        print("3. Run performance evaluation")
        print("4. Document the integration in your dissertation")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("Please check the server logs and fix any issues")

if __name__ == "__main__":
    main()
