#!/usr/bin/env python3
"""
Script to download and verify ISOT dataset for fake news detection
This implements the immediate actions from the model performance plan
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dataset_manager import dataset_manager
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_isot():
    """Download ISOT dataset"""
    print("=" * 60)
    print("ISOT Dataset Downloader")
    print("=" * 60)
    print("\nThis script will download the ISOT Fake News Dataset")
    print("Size: ~150MB compressed, 44,898 articles total")
    print("Source: University of Victoria")
    print("=" * 60)
    
    # Check if already exists
    dataset_dir = Path("datasets/isot")
    if (dataset_dir / "Fake.csv").exists() and (dataset_dir / "True.csv").exists():
        print("\n✅ ISOT dataset already exists!")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return True
    
    print("\nStarting download...")
    success = dataset_manager.download_isot_dataset(use_kaggle=False)
    
    if success:
        print("\n✅ Download successful!")
        return True
    else:
        print("\n❌ Download failed!")
        print("\nAlternative download methods:")
        print("1. Manual download:")
        print("   wget https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/ISOT_Fake_News_Dataset.zip")
        print("   unzip ISOT_Fake_News_Dataset.zip -d datasets/isot/")
        print("\n2. Kaggle (requires API key):")
        print("   kaggle datasets download -d csmalarkodi/isot-fake-news-dataset")
        return False

def verify_isot():
    """Verify ISOT dataset can be loaded"""
    print("\n" + "=" * 60)
    print("Verifying ISOT Dataset")
    print("=" * 60)
    
    # Try to load the dataset
    df = dataset_manager.load_isot_dataset()
    
    if df is None:
        print("❌ Failed to load ISOT dataset")
        return False
    
    print(f"\n✅ Successfully loaded ISOT dataset!")
    print(f"Total records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Show label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel Distribution:")
    print(f"  Real articles (label=0): {label_counts.get(0, 0):,}")
    print(f"  Fake articles (label=1): {label_counts.get(1, 0):,}")
    
    # Show sample records
    print("\nSample records:")
    print("-" * 40)
    for i, row in df.head(3).iterrows():
        print(f"\nRecord {i+1}:")
        print(f"  Label: {row['label']} ({row.get('label_text', 'N/A')})")
        print(f"  Statement: {row['statement'][:100]}...")
        if 'title' in df.columns:
            print(f"  Title: {row.get('title', 'N/A')[:80]}...")
    
    # Verify compatibility with preprocessing
    print("\n" + "=" * 60)
    print("Testing Preprocessing Compatibility")
    print("=" * 60)
    
    try:
        from app.preprocessing import preprocessor
        
        # Test preprocessing on a sample
        sample_df = df.head(10).copy()
        processed_df = preprocessor.preprocess_dataset(
            sample_df,
            text_column='statement',
            label_column='label'
        )
        
        print(f"✅ Preprocessing successful!")
        print(f"Original columns: {list(sample_df.columns)}")
        print(f"Processed columns: {list(processed_df.columns)}")
        
        # Check for expected features
        expected_features = ['word_count', 'sentence_count', 'sentiment_polarity']
        found_features = [f for f in expected_features if f in processed_df.columns]
        print(f"Features added: {found_features}")
        
    except Exception as e:
        print(f"⚠️ Preprocessing test failed: {e}")
        print("This is not critical - the dataset can still be used")
    
    return True

def quick_statistics():
    """Show quick statistics about the dataset"""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    df = dataset_manager.load_isot_dataset()
    if df is None:
        print("Dataset not loaded")
        return
    
    # Text length statistics
    df['text_length'] = df['statement'].str.len()
    df['word_count'] = df['statement'].str.split().str.len()
    
    print("\nText Statistics:")
    print(f"  Average text length: {df['text_length'].mean():.0f} characters")
    print(f"  Average word count: {df['word_count'].mean():.0f} words")
    print(f"  Min text length: {df['text_length'].min()} characters")
    print(f"  Max text length: {df['text_length'].max():,} characters")
    
    if 'subject' in df.columns:
        print("\nTop 5 Subjects:")
        for subject, count in df['subject'].value_counts().head(5).items():
            print(f"  {subject}: {count:,}")
    
    print("\n✅ ISOT dataset is ready for training!")
    print("\nNext steps:")
    print("1. Use pre-trained model for immediate high accuracy")
    print("2. Fine-tune on ISOT for domain-specific performance")
    print("3. Run evaluation to verify performance meets targets")

def main():
    """Main execution"""
    print("\n🚀 ISOT Dataset Setup for Fake News Detection")
    print("=" * 60)
    
    # Step 1: Download
    if not download_isot():
        print("\n❌ Setup failed at download step")
        return 1
    
    # Step 2: Verify
    if not verify_isot():
        print("\n❌ Setup failed at verification step")
        return 1
    
    # Step 3: Statistics
    quick_statistics()
    
    print("\n" + "=" * 60)
    print("✅ ISOT Dataset Setup Complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Start training: python scripts/train_universal.py --dataset isot")
    print("2. Test pre-trained model: python scripts/test_pretrained.py")
    print("3. Run full pipeline: python scripts/run_dissertation_pipeline.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
