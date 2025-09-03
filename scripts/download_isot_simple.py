#!/usr/bin/env python3
"""
Simple ISOT Dataset Downloader
Uses HuggingFace datasets to download ISOT dataset directly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datasets import load_dataset
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_isot_via_huggingface():
    """Download ISOT dataset using HuggingFace datasets"""
    
    print("=" * 60)
    print("🚀 ISOT Dataset Download via HuggingFace")
    print("=" * 60)
    
    try:
        print("⏳ Loading ISOT dataset from HuggingFace...")
        
        # Load the dataset
        dataset = load_dataset("isot/fake-news")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train: {len(dataset['train'])} samples")
        print(f"   Test: {len(dataset['test'])} samples")
        
        # Convert to pandas DataFrames
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        # Combine train and test
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"   Total: {len(combined_df)} samples")
        
        # Create datasets directory
        datasets_dir = Path("datasets/isot")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into fake and real news
        fake_news = combined_df[combined_df['label'] == 1].copy()
        real_news = combined_df[combined_df['label'] == 0].copy()
        
        # Save as CSV files in ISOT format
        fake_news[['text']].to_csv(datasets_dir / "Fake.csv", index=False)
        real_news[['text']].to_csv(datasets_dir / "True.csv", index=False)
        
        print(f"✅ Files saved to {datasets_dir}")
        print(f"   Fake.csv: {len(fake_news)} articles")
        print(f"   True.csv: {len(real_news)} articles")
        
        # Create metadata
        metadata = {
            "downloaded_at": datetime.now().isoformat(),
            "source": "HuggingFace Datasets",
            "dataset_name": "isot/fake-news",
            "total_samples": len(combined_df),
            "fake_samples": len(fake_news),
            "real_samples": len(real_news),
            "files": ["Fake.csv", "True.csv"]
        }
        
        # Save metadata
        import json
        with open(datasets_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Metadata saved")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download ISOT dataset: {e}")
        print(f"❌ Download failed: {e}")
        return False

def verify_download():
    """Verify the downloaded files"""
    
    print("\n" + "=" * 60)
    print("🔍 Verifying Download")
    print("=" * 60)
    
    datasets_dir = Path("datasets/isot")
    
    fake_path = datasets_dir / "Fake.csv"
    true_path = datasets_dir / "True.csv"
    metadata_path = datasets_dir / "metadata.json"
    
    if not fake_path.exists():
        print("❌ Fake.csv not found")
        return False
    
    if not true_path.exists():
        print("❌ True.csv not found")
        return False
    
    if not metadata_path.exists():
        print("❌ metadata.json not found")
        return False
    
    # Check file contents
    try:
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        print(f"✅ Fake.csv: {len(fake_df)} articles")
        print(f"✅ True.csv: {len(true_df)} articles")
        
        # Show sample
        print(f"\n📄 Sample fake news:")
        print(f"   {fake_df['text'].iloc[0][:100]}...")
        
        print(f"\n📄 Sample real news:")
        print(f"   {true_df['text'].iloc[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return False

def main():
    """Main function"""
    
    print("🚀 Simple ISOT Dataset Downloader")
    print("=" * 60)
    print("This script downloads the ISOT fake news dataset")
    print("using HuggingFace datasets (no manual download needed)")
    print("=" * 60)
    
    # Download the dataset
    success = download_isot_via_huggingface()
    
    if success:
        # Verify the download
        verify_success = verify_download()
        
        if verify_success:
            print("\n" + "🎉" * 20)
            print("🎉 SUCCESS! ISOT dataset downloaded and verified")
            print("🎉 You can now use it for training and evaluation")
            print("🎉" * 20)
            
            print("\n📋 Next Steps:")
            print("1. Test the model: python scripts/test_pretrained.py")
            print("2. Run evaluation: python scripts/test_phase1_integration.py")
            print("3. Integrate into API: Update app/main.py")
        else:
            print("\n⚠️ Download completed but verification failed")
    else:
        print("\n❌ Download failed")
        print("\nAlternative options:")
        print("1. Use existing datasets (LIAR, PolitiFact)")
        print("2. Try manual download from Kaggle")
        print("3. Use the model without ISOT dataset")

if __name__ == "__main__":
    main()
