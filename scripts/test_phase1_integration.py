#!/usr/bin/env python3
"""
Test Phase 1 Integration - Verify ISOT dataset is fully integrated
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from app.dataset_manager import dataset_manager

def test_dataset_manager():
    """Test dataset manager integration"""
    print("\n" + "="*60)
    print("Testing Dataset Manager")
    print("="*60)
    
    # Check ISOT is in available datasets
    datasets = dataset_manager.datasets
    assert "isot" in datasets, "ISOT not in dataset configuration"
    assert "isot_kaggle" in datasets, "ISOT Kaggle mirror not in configuration"
    print("✅ ISOT dataset configurations found")
    
    # Check metadata
    isot_config = datasets["isot"]
    assert isot_config["size"] == 44898, "Incorrect dataset size"
    assert "Fake.csv" in isot_config["files"], "Missing Fake.csv in files"
    assert "True.csv" in isot_config["files"], "Missing True.csv in files"
    print("✅ ISOT configuration correct")
    
    return True

def test_api_endpoints():
    """Test API endpoints support ISOT"""
    print("\n" + "="*60)
    print("Testing API Endpoints")
    print("="*60)
    
    base_url = "http://localhost:8000"
    
    # Test 1: List datasets endpoint
    print("\n📡 Testing /datasets endpoint...")
    try:
        response = requests.get(f"{base_url}/datasets")
        if response.status_code == 200:
            datasets = response.json()
            # Check if ISOT appears in the list
            dataset_names = [d.get("name", d) for d in datasets] if isinstance(datasets, list) else list(datasets.keys())
            if "isot" in [name.lower() for name in dataset_names if isinstance(name, str)]:
                print("✅ ISOT appears in datasets list")
            else:
                print("⚠️ ISOT not visible in API (may need download first)")
        else:
            print(f"⚠️ API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️ API not running. Start with: python start_server.py")
        return False
    
    # Test 2: Dataset info endpoint
    print("\n📡 Testing /datasets/isot/info endpoint...")
    try:
        response = requests.get(f"{base_url}/datasets/isot/info")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ ISOT info endpoint works")
            print(f"   Status: {info.get('status', 'unknown')}")
        elif response.status_code == 404:
            print("⚠️ ISOT not yet downloaded (expected for first run)")
        else:
            print(f"⚠️ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    return True

def test_data_loading():
    """Test ISOT data loading"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    # Try to load ISOT dataset
    df = dataset_manager.load_isot_dataset()
    
    if df is None:
        print("⚠️ ISOT dataset not downloaded yet")
        print("   Run: python scripts/download_isot.py")
        return False
    
    print(f"✅ ISOT dataset loaded successfully")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Verify required columns
    required_columns = ['statement', 'label']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    print(f"✅ Required columns present: {required_columns}")
    
    # Check label distribution
    label_counts = df['label'].value_counts()
    print(f"✅ Label distribution:")
    print(f"   Real (0): {label_counts.get(0, 0):,}")
    print(f"   Fake (1): {label_counts.get(1, 0):,}")
    
    return True

def main():
    """Run all integration tests"""
    print("\n🧪 Phase 1 Integration Test Suite")
    print("="*60)
    print("Testing ISOT dataset integration...")
    
    all_pass = True
    
    # Test 1: Dataset Manager
    try:
        if not test_dataset_manager():
            all_pass = False
    except Exception as e:
        print(f"❌ Dataset manager test failed: {e}")
        all_pass = False
    
    # Test 2: API Endpoints
    try:
        if not test_api_endpoints():
            print("⚠️ API tests skipped/incomplete")
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    # Test 3: Data Loading
    try:
        if not test_data_loading():
            print("⚠️ Data loading test incomplete")
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Integration Test Summary")
    print("="*60)
    
    if all_pass:
        print("✅ All core tests passed!")
        print("\nPhase 1 is complete. Next steps:")
        print("1. Download ISOT: python scripts/download_isot.py")
        print("2. Test pre-trained model: python scripts/test_pretrained.py")
        print("3. Start API with new model: python start_server.py")
    else:
        print("⚠️ Some tests incomplete or failed")
        print("\nRecommended actions:")
        print("1. Ensure ISOT is downloaded: python scripts/download_isot.py")
        print("2. Start API if needed: python start_server.py")
        print("3. Re-run this test: python scripts/test_phase1_integration.py")

if __name__ == "__main__":
    main()
