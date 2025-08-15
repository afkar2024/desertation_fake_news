#!/usr/bin/env python3
"""
Setup script for initializing datasets after fresh repository clone
"""
import sys
import os
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def main():
    """Main setup function for datasets"""
    print("🚀 Setting up datasets for Adaptive Fake News Detector")
    print("=" * 60)
    
    try:
        from app.dataset_manager import dataset_manager
        
        print("📊 Current dataset status:")
        datasets = dataset_manager.list_datasets()
        
        if not datasets:
            print("❌ No datasets available")
            return False
        
        for name, info in datasets.items():
            status = info.get('status', 'unknown')
            status_icon = "✅" if status == "downloaded" else "⏳" if status == "not_downloaded" else "⚠️"
            print(f"   {status_icon} {name}: {status}")
        
        print("\n🔄 Downloading available datasets...")
        
        # Download datasets that can be automatically downloaded
        success_count = 0
        
        # PolitiFact can be generated automatically
        print("📥 Downloading PolitiFact dataset...")
        if dataset_manager.fetch_politifact_data():
            print("✅ PolitiFact dataset downloaded")
            success_count += 1
        else:
            print("❌ Failed to download PolitiFact dataset")
        
        # LIAR dataset requires internet download
        print("\n📥 LIAR dataset requires internet download...")
        print("   This will download ~50MB from the internet")
        download_liar = input("   Download LIAR dataset now? (y/N): ").strip().lower()
        
        if download_liar == 'y':
            if dataset_manager.download_liar_dataset():
                print("✅ LIAR dataset downloaded")
                success_count += 1
            else:
                print("❌ Failed to download LIAR dataset")
        else:
            print("⏭️  Skipping LIAR dataset download")
        
        # FakeNewsNet requires manual setup
        print("\n📋 FakeNewsNet dataset requires manual setup:")
        print("   1. Clone: git clone https://github.com/KaiDMML/FakeNewsNet.git")
        print("   2. Follow their README for data download")
        print("   3. Place data in datasets/fakenewsnet/")
        
        print(f"\n🎉 Setup complete! {success_count} datasets downloaded.")
        print("\n💡 To download all datasets later, run:")
        print("   python download_and_preprocess.py --download")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
