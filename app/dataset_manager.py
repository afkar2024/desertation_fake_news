"""
Dataset Manager for Fake News Detection
Handles downloading and managing datasets from various sources as mentioned in the pilot report.
"""

import os
import json
import pandas as pd
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import hashlib
from urllib.parse import urlparse

from .config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages downloading and caching of datasets for fake news detection"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "liar": {
                "name": "LIAR Dataset",
                "url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
                "description": "12,836 political statements labeled across six veracity categories",
                "files": ["train.tsv", "test.tsv", "valid.tsv"],
                "format": "tsv"
            },
            "fakenewsnet": {
                "name": "FakeNewsNet",
                "url": "https://github.com/KaiDMML/FakeNewsNet",
                "description": "Multimodal corpus combining text, user profiles, and network propagation features",
                "files": [],
                "format": "json"
            },
            "politifact": {
                "name": "PolitiFact",
                "url": "custom",  # Will be fetched via API
                "description": "Fact-checked articles with granular veracity ratings",
                "files": ["politifact_data.json"],
                "format": "json"
            }
        }
        
        self.metadata_file = self.data_dir / "datasets_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load dataset metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """Save dataset metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download a file from URL with progress tracking"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownloading {filepath.name}: {progress:.1f}%", end='')
            
            print(f"\nDownloaded: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract zip or tar archive"""
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.warning(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            logger.info(f"Extracted: {archive_path} to {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {e}")
            return False
    
    def download_liar_dataset(self) -> bool:
        """Download and extract LIAR dataset"""
        dataset_name = "liar"
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        
        logger.info(f"Downloading {dataset_info['name']}...")
        
        # Download the zip file
        zip_path = dataset_dir / "liar_dataset.zip"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.download_file(dataset_info["url"], zip_path):
            return False
        
        # Extract the archive
        if not self.extract_archive(zip_path, dataset_dir):
            return False
        
        # Update metadata
        self.metadata[dataset_name] = {
            "downloaded_at": datetime.now().isoformat(),
            "source_url": dataset_info["url"],
            "local_path": str(dataset_dir),
            "files": dataset_info["files"],
            "hash": self.get_file_hash(zip_path)
        }
        
        self.save_metadata()
        logger.info(f"Successfully downloaded {dataset_info['name']}")
        return True
    
    def download_fakenewsnet_dataset(self) -> bool:
        """Download FakeNewsNet dataset from GitHub"""
        dataset_name = "fakenewsnet"
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        
        logger.info(f"Downloading {dataset_info['name']}...")
        
        # For now, we'll create a placeholder and instructions
        # In a real implementation, you'd clone the GitHub repo
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        instructions_file = dataset_dir / "README.md"
        instructions = f"""
# FakeNewsNet Dataset

This dataset should be downloaded from: {dataset_info['url']}

To download manually:
1. Clone the repository: `git clone https://github.com/KaiDMML/FakeNewsNet.git`
2. Follow the instructions in their README to download the actual data files
3. Place the data files in this directory: {dataset_dir}

The dataset includes:
- PolitiFact fake news dataset
- GossipCop fake news dataset
- User profiles and social network features
- News content and metadata
"""
        
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        # Update metadata
        self.metadata[dataset_name] = {
            "downloaded_at": datetime.now().isoformat(),
            "source_url": dataset_info["url"],
            "local_path": str(dataset_dir),
            "status": "manual_download_required",
            "instructions": str(instructions_file)
        }
        
        self.save_metadata()
        logger.info(f"Created instructions for {dataset_info['name']}")
        return True
    
    def fetch_politifact_data(self, limit: int = 1000) -> bool:
        """Fetch PolitiFact data (placeholder - would need actual API access)"""
        dataset_name = "politifact"
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Fetching {dataset_info['name']} data...")
        
        # Placeholder data structure
        sample_data = {
            "articles": [
                {
                    "id": f"politifact_{i}",
                    "statement": f"Sample political statement {i}",
                    "label": ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"][i % 6],
                    "subject": f"Subject {i}",
                    "speaker": f"Speaker {i}",
                    "context": f"Context for statement {i}",
                    "date": "2024-01-01",
                    "source_url": f"https://www.politifact.com/statement/{i}/"
                }
                for i in range(min(limit, 100))  # Create sample data
            ],
            "metadata": {
                "total_count": min(limit, 100),
                "fetched_at": datetime.now().isoformat(),
                "labels": ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"]
            }
        }
        
        # Save sample data
        data_file = dataset_dir / "politifact_data.json"
        with open(data_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        # Update metadata
        self.metadata[dataset_name] = {
            "downloaded_at": datetime.now().isoformat(),
            "source_url": "PolitiFact API",
            "local_path": str(dataset_dir),
            "files": ["politifact_data.json"],
            "record_count": len(sample_data["articles"])
        }
        
        self.save_metadata()
        logger.info(f"Fetched {len(sample_data['articles'])} records from {dataset_info['name']}")
        return True
    
    def download_all_datasets(self) -> Dict[str, bool]:
        """Download all available datasets"""
        results = {}
        
        logger.info("Starting download of all datasets...")
        
        # Download LIAR dataset
        results["liar"] = self.download_liar_dataset()
        
        # Download FakeNewsNet (instructions)
        results["fakenewsnet"] = self.download_fakenewsnet_dataset()
        
        # Fetch PolitiFact data
        results["politifact"] = self.fetch_politifact_data()
        
        logger.info(f"Dataset download results: {results}")
        return results
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a specific dataset"""
        return self.metadata.get(dataset_name)
    
    def list_datasets(self) -> Dict[str, Dict]:
        """List all available datasets with their metadata"""
        # If no metadata exists, return available dataset info
        if not self.metadata:
            available_datasets = {}
            for name, info in self.datasets.items():
                available_datasets[name] = {
                    "name": info["name"],
                    "description": info["description"],
                    "status": "not_downloaded",
                    "available": True,
                    "download_required": True
                }
            return available_datasets
        
        # Return existing metadata with additional info
        result = {}
        for name, info in self.datasets.items():
            if name in self.metadata:
                # Dataset exists
                result[name] = {
                    **self.metadata[name],
                    "name": info["name"],
                    "description": info["description"],
                    "status": self.metadata[name].get("status", "downloaded"),
                    "available": True,
                    "download_required": False
                }
            else:
                # Dataset not downloaded yet
                result[name] = {
                    "name": info["name"],
                    "description": info["description"],
                    "status": "not_downloaded",
                    "available": True,
                    "download_required": True
                }
        
        return result
    
    def get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """Get the local path for a dataset"""
        if dataset_name in self.metadata:
            return Path(self.metadata[dataset_name]["local_path"])
        return None
    
    def load_liar_dataset(self) -> Optional[pd.DataFrame]:
        """Load LIAR dataset into pandas DataFrame"""
        dataset_path = self.get_dataset_path("liar")
        if not dataset_path or not dataset_path.exists():
            logger.error("LIAR dataset not found. Please download it first.")
            return None
        
        try:
            # LIAR dataset has TSV files
            train_file = dataset_path / "train.tsv"
            test_file = dataset_path / "test.tsv"
            valid_file = dataset_path / "valid.tsv"
            
            columns = [
                "id", "label", "statement", "subject", "speaker", "job_title",
                "state_info", "party_affiliation", "barely_true_counts",
                "false_counts", "half_true_counts", "mostly_true_counts",
                "pants_on_fire_counts", "context"
            ]
            
            dataframes = []
            for file_path in [train_file, test_file, valid_file]:
                if file_path.exists():
                    df = pd.read_csv(file_path, sep='\t', names=columns, header=None)
                    df['split'] = file_path.stem
                    dataframes.append(df)
            
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                logger.info(f"Loaded LIAR dataset with {len(combined_df)} records")
                return combined_df
            
        except Exception as e:
            logger.error(f"Error loading LIAR dataset: {e}")
        
        return None
    
    def load_politifact_dataset(self) -> Optional[pd.DataFrame]:
        """Load PolitiFact dataset into pandas DataFrame"""
        dataset_path = self.get_dataset_path("politifact")
        if not dataset_path or not dataset_path.exists():
            logger.error("PolitiFact dataset not found. Please fetch it first.")
            return None
        
        try:
            data_file = dataset_path / "politifact_data.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data["articles"])
                logger.info(f"Loaded PolitiFact dataset with {len(df)} records")
                return df
            
        except Exception as e:
            logger.error(f"Error loading PolitiFact dataset: {e}")
        
        return None


# Global dataset manager instance
dataset_manager = DatasetManager() 