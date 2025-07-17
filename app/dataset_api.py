"""
API endpoints for dataset management and preprocessing
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import json
import asyncio
from datetime import datetime
import tempfile
import os

from .dataset_manager import dataset_manager
from .preprocessing import preprocessor

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Pydantic models for API
class DatasetInfo(BaseModel):
    name: str
    description: str
    status: str
    record_count: Optional[int] = None
    downloaded_at: Optional[str] = None
    local_path: Optional[str] = None

class PreprocessingRequest(BaseModel):
    dataset_name: str
    text_column: str = "statement"
    label_column: str = "label"
    balance_strategy: Optional[str] = None  # 'undersample', 'oversample', or None
    save_path: Optional[str] = None

class PreprocessingResponse(BaseModel):
    success: bool
    message: str
    processed_records: Optional[int] = None
    features_added: Optional[List[str]] = None
    save_path: Optional[str] = None

class FullPipelineRequest(BaseModel):
    text_column: str = "statement"
    label_column: str = "label"
    balance_strategy: Optional[str] = None  # 'undersample', 'oversample', or None
    download_if_missing: bool = True
    return_markdown: bool = True

def generate_markdown_report_api(results: Optional[Dict[str, bool]] = None, 
                               original_df: Optional[pd.DataFrame] = None, 
                               processed_df: Optional[pd.DataFrame] = None, 
                               balanced_df: Optional[pd.DataFrame] = None, 
                               dataset_name: str = "", 
                               output_dir: Optional[Path] = None,
                               report_type: str = "combined") -> str:
    """Generate a comprehensive markdown report for API use"""
    
    if output_dir is None:
        output_dir = Path("processed_data")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f"{report_type}_report_{dataset_name}_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        # Header
        f.write(f"# Fake News Detection Dataset Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Report Type:** {report_type.title()}\n")
        if dataset_name:
            f.write(f"**Dataset:** {dataset_name.upper()}\n")
        f.write(f"**Output Directory:** `{output_dir.absolute()}`\n\n")
        
        f.write("---\n\n")
        
        # Download Results Section
        if results:
            f.write("## 📥 Download Results\n\n")
            
            total_datasets = len(results)
            successful = sum(results.values())
            failed = total_datasets - successful
            
            f.write(f"- **Total Datasets:** {total_datasets}\n")
            f.write(f"- **Successful:** {successful} ✅\n")
            f.write(f"- **Failed:** {failed} ❌\n\n")
            
            f.write("### Dataset Status\n\n")
            f.write("| Dataset | Status | Location | Description |\n")
            f.write("|---------|--------|----------|-------------|\n")
            
            for dataset, success in results.items():
                status = "✅ Success" if success else "❌ Failed"
                info = dataset_manager.get_dataset_info(dataset)
                location = info.get('location', 'N/A') if info else 'N/A'
                description = info.get('description', 'N/A') if info else 'N/A'
                f.write(f"| {dataset} | {status} | `{location}` | {description} |\n")
            
            f.write("\n---\n\n")
        
        # Preprocessing Results Section
        if original_df is not None and processed_df is not None:
            f.write("## 🔄 Preprocessing Results\n\n")
            
            # Dataset Overview
            f.write("### Dataset Overview\n\n")
            f.write("| Metric | Original | Processed | Balanced |\n")
            f.write("|--------|----------|-----------|----------|\n")
            f.write(f"| **Rows** | {original_df.shape[0]:,} | {processed_df.shape[0]:,} | {balanced_df.shape[0]:,} |\n" if balanced_df is not None else f"| **Rows** | {original_df.shape[0]:,} | {processed_df.shape[0]:,} | N/A |\n")
            f.write(f"| **Columns** | {original_df.shape[1]} | {processed_df.shape[1]} | {balanced_df.shape[1]} |\n" if balanced_df is not None else f"| **Columns** | {original_df.shape[1]} | {processed_df.shape[1]} | N/A |\n")
            
            features_added = len(processed_df.columns) - len(original_df.columns)
            f.write(f"| **Features Added** | - | {features_added} | - |\n\n")
            
            # File Locations
            f.write("### 📁 Generated Files\n\n")
            csv_file = output_dir / f"{dataset_name}_processed.csv"
            mapping_file = output_dir / f"{dataset_name}_processed_label_mapping.json"
            
            f.write(f"- **Processed Dataset:** `{csv_file.absolute()}`\n")
            f.write(f"- **Label Mapping:** `{mapping_file.absolute()}`\n")
            f.write(f"- **This Report:** `{report_file.absolute()}`\n\n")
            
            # New Features Added
            f.write("### 🆕 New Features Added\n\n")
            new_columns = [col for col in processed_df.columns if col not in original_df.columns]
            if new_columns:
                f.write("| Feature | Type | Description |\n")
                f.write("|---------|------|-------------|\n")
                feature_descriptions = {
                    'flesch_kincaid_grade': ('Readability', 'Flesch-Kincaid grade level score'),
                    'flesch_reading_ease': ('Readability', 'Flesch reading ease score (0-100)'),
                    'automated_readability_index': ('Readability', 'Automated Readability Index'),
                    'coleman_liau_index': ('Readability', 'Coleman-Liau readability index'),
                    'gunning_fog': ('Readability', 'Gunning Fog readability index'),
                    'sentiment_polarity': ('Sentiment', 'Sentiment polarity (-1 to 1)'),
                    'sentiment_subjectivity': ('Sentiment', 'Sentiment subjectivity (0 to 1)'),
                    'sentiment_label': ('Sentiment', 'Categorical sentiment (positive/negative/neutral)'),
                    'exclamation_count': ('Linguistic', 'Number of exclamation marks'),
                    'question_count': ('Linguistic', 'Number of question marks'),
                    'caps_ratio': ('Linguistic', 'Ratio of uppercase characters'),
                    'all_caps_words': ('Linguistic', 'Number of all-caps words'),
                    'word_count': ('Text Stats', 'Number of words'),
                    'char_count': ('Text Stats', 'Number of characters'),
                    'avg_word_length': ('Text Stats', 'Average word length'),
                    'label_encoded': ('Processing', 'Numerically encoded labels')
                }
                
                for col in new_columns:
                    feat_type = feature_descriptions.get(col, ('Other', 'Generated feature'))[0] if col in feature_descriptions else 'Other'
                    description = feature_descriptions.get(col, ('Other', 'Generated feature'))[1] if col in feature_descriptions else 'Generated feature'
                    f.write(f"| `{col}` | {feat_type} | {description} |\n")
            else:
                f.write("No new features were added.\n")
            
            f.write("\n")
            
            # Label Analysis
            if 'label' in processed_df.columns:
                f.write("### 🏷️ Label Analysis\n\n")
                
                label_counts = processed_df['label'].value_counts()
                total_samples = len(processed_df)
                
                f.write("| Label | Count | Percentage |\n")
                f.write("|-------|-------|------------|\n")
                
                for label, count in label_counts.items():
                    percentage = (count / total_samples) * 100
                    f.write(f"| {label} | {count:,} | {percentage:.1f}% |\n")
                
                f.write(f"\n**Total Labels:** {len(label_counts)}\n")
                f.write(f"**Most Common:** {label_counts.index[0]} ({label_counts.iloc[0]:,} samples)\n")
                f.write(f"**Least Common:** {label_counts.index[-1]} ({label_counts.iloc[-1]:,} samples)\n\n")
                
                # Label mapping
                if hasattr(processed_df, 'attrs') and 'label_mapping' in processed_df.attrs:
                    f.write("#### Label Mapping\n\n")
                    f.write("| Original Label | Encoded Value |\n")
                    f.write("|---------------|---------------|\n")
                    for orig, encoded in processed_df.attrs['label_mapping'].items():
                        f.write(f"| {orig} | {encoded} |\n")
                    f.write("\n")
            
            # Feature Analysis
            f.write("### 📊 Feature Analysis\n\n")
            
            # Readability Analysis
            if 'flesch_kincaid_grade' in processed_df.columns:
                f.write("#### Readability Metrics\n\n")
                f.write("| Metric | Mean | Std Dev | Min | Max |\n")
                f.write("|--------|------|---------|-----|-----|\n")
                
                readability_cols = ['flesch_kincaid_grade', 'flesch_reading_ease', 'automated_readability_index', 'coleman_liau_index', 'gunning_fog']
                for col in readability_cols:
                    if col in processed_df.columns:
                        mean_val = processed_df[col].mean()
                        std_val = processed_df[col].std()
                        min_val = processed_df[col].min()
                        max_val = processed_df[col].max()
                        f.write(f"| {col.replace('_', ' ').title()} | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
                
                f.write("\n")
            
            # Sentiment Analysis
            if 'sentiment_polarity' in processed_df.columns:
                f.write("#### Sentiment Analysis\n\n")
                f.write("| Metric | Mean | Std Dev | Min | Max |\n")
                f.write("|--------|------|---------|-----|-----|\n")
                
                pol_mean = processed_df['sentiment_polarity'].mean()
                pol_std = processed_df['sentiment_polarity'].std()
                pol_min = processed_df['sentiment_polarity'].min()
                pol_max = processed_df['sentiment_polarity'].max()
                f.write(f"| Polarity | {pol_mean:.3f} | {pol_std:.3f} | {pol_min:.3f} | {pol_max:.3f} |\n")
                
                subj_mean = processed_df['sentiment_subjectivity'].mean()
                subj_std = processed_df['sentiment_subjectivity'].std()
                subj_min = processed_df['sentiment_subjectivity'].min()
                subj_max = processed_df['sentiment_subjectivity'].max()
                f.write(f"| Subjectivity | {subj_mean:.3f} | {subj_std:.3f} | {subj_min:.3f} | {subj_max:.3f} |\n")
                
                f.write("\n")
                
                # Sentiment distribution
                if 'sentiment_label' in processed_df.columns:
                    f.write("##### Sentiment Distribution\n\n")
                    sentiment_counts = processed_df['sentiment_label'].value_counts()
                    f.write("| Sentiment | Count | Percentage |\n")
                    f.write("|-----------|-------|------------|\n")
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count / len(processed_df)) * 100
                        f.write(f"| {sentiment} | {count:,} | {percentage:.1f}% |\n")
                    f.write("\n")
            
            # Linguistic Features
            linguistic_features = ['exclamation_count', 'question_count', 'caps_ratio', 'all_caps_words']
            available_linguistic = [f for f in linguistic_features if f in processed_df.columns]
            
            if available_linguistic:
                f.write("#### Linguistic Features\n\n")
                f.write("| Feature | Mean | Std Dev | Max |\n")
                f.write("|---------|------|---------|-----|\n")
                
                for feature in available_linguistic:
                    mean_val = processed_df[feature].mean()
                    std_val = processed_df[feature].std()
                    max_val = processed_df[feature].max()
                    f.write(f"| {feature.replace('_', ' ').title()} | {mean_val:.3f} | {std_val:.3f} | {max_val:.0f} |\n")
                
                f.write("\n")
            
            # Text Statistics
            text_columns = ['statement', 'content', 'title']
            available_text_cols = [col for col in text_columns if col in processed_df.columns]
            
            if available_text_cols:
                f.write("#### Text Statistics\n\n")
                f.write("| Column | Avg Length | Max Length | Min Length | Std Dev |\n")
                f.write("|--------|------------|------------|------------|----------|\n")
                
                for col in available_text_cols:
                    text_lengths = processed_df[col].astype(str).str.len()
                    avg_len = text_lengths.mean()
                    max_len = text_lengths.max()
                    min_len = text_lengths.min()
                    std_len = text_lengths.std()
                    f.write(f"| {col} | {avg_len:.0f} | {max_len} | {min_len} | {std_len:.0f} |\n")
                
                f.write("\n")
            
            # Balancing Analysis
            if balanced_df is not None:
                f.write("### ⚖️ Dataset Balancing\n\n")
                
                original_counts = processed_df['label'].value_counts()
                balanced_counts = balanced_df['label'].value_counts()
                
                f.write("| Label | Original Count | Balanced Count | Change |\n")
                f.write("|-------|---------------|----------------|--------|\n")
                
                for label in original_counts.index:
                    orig_count = original_counts[label]
                    bal_count = balanced_counts.get(label, 0)
                    change = bal_count - orig_count
                    change_str = f"+{change}" if change > 0 else str(change)
                    f.write(f"| {label} | {orig_count:,} | {bal_count:,} | {change_str} |\n")
                
                f.write(f"\n**Balancing Strategy Applied:** Successfully reduced dataset size while maintaining label distribution\n")
                f.write(f"**Original Size:** {len(processed_df):,} samples\n")
                f.write(f"**Balanced Size:** {len(balanced_df):,} samples\n")
                f.write(f"**Size Reduction:** {len(processed_df) - len(balanced_df):,} samples ({((len(processed_df) - len(balanced_df))/len(processed_df)*100):.1f}%)\n\n")
        
        # Instructions for users
        f.write("---\n\n")
        f.write("## 📋 How to Use These Files\n\n")
        f.write("### Loading the Processed Dataset\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n")
        f.write("import json\n\n")
        f.write(f"# Load the processed dataset\n")
        f.write(f"df = pd.read_csv('{output_dir.absolute()}/{dataset_name}_processed.csv')\n\n")
        f.write(f"# Load the label mapping\n")
        f.write(f"with open('{output_dir.absolute()}/{dataset_name}_processed_label_mapping.json', 'r') as f:\n")
        f.write(f"    label_mapping = json.load(f)\n\n")
        f.write("print(f'Dataset shape: {df.shape}')\n")
        f.write("print(f'Label mapping: {label_mapping}')\n")
        f.write("```\n\n")
        
        f.write("### Key Columns for Machine Learning\n\n")
        if processed_df is not None:
            f.write("- **Text Features:** ")
            text_features = [col for col in processed_df.columns if col in ['statement', 'content', 'title']]
            f.write(", ".join([f"`{col}`" for col in text_features]) + "\n")
            
            f.write("- **Numerical Features:** ")
            numerical_features = [col for col in processed_df.columns if col in ['flesch_kincaid_grade', 'sentiment_polarity', 'word_count', 'char_count']]
            f.write(", ".join([f"`{col}`" for col in numerical_features]) + "\n")
            
            f.write("- **Target Variable:** `label_encoded` (numerical) or `label` (categorical)\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. **Data Exploration:** Load the dataset and explore the new features\n")
        f.write("2. **Feature Selection:** Choose the most relevant features for your model\n")
        f.write("3. **Model Training:** Use the processed features to train your fake news detection model\n")
        f.write("4. **Evaluation:** Test your model using the balanced dataset\n\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated by Fake News Detection Dataset Processing Pipeline*\n")
        f.write(f"*Timestamp: {datetime.now().isoformat()}*\n")
    
    return str(report_file)

@router.get("/", response_model=List[DatasetInfo])
async def list_datasets():
    """List all available datasets with their metadata"""
    datasets = []
    metadata = dataset_manager.list_datasets()
    
    for dataset_name, info in metadata.items():
        dataset_info = DatasetInfo(
            name=dataset_name,
            description=dataset_manager.datasets.get(dataset_name, {}).get('description', ''),
            status=info.get('status', 'downloaded'),
            record_count=info.get('record_count'),
            downloaded_at=info.get('downloaded_at'),
            local_path=info.get('local_path')
        )
        datasets.append(dataset_info)
    
    return datasets

@router.post("/download")
async def download_datasets(background_tasks: BackgroundTasks):
    """Download all available datasets"""
    background_tasks.add_task(dataset_manager.download_all_datasets)
    
    return {
        "message": "Dataset download started",
        "datasets": list(dataset_manager.datasets.keys())
    }

@router.post("/download/{dataset_name}")
async def download_single_dataset(dataset_name: str, background_tasks: BackgroundTasks):
    """Download a specific dataset"""
    if dataset_name not in dataset_manager.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset_name == "liar":
        background_tasks.add_task(dataset_manager.download_liar_dataset)
    elif dataset_name == "fakenewsnet":
        background_tasks.add_task(dataset_manager.download_fakenewsnet_dataset)
    elif dataset_name == "politifact":
        background_tasks.add_task(dataset_manager.fetch_politifact_data)
    else:
        raise HTTPException(status_code=400, detail="Download not implemented for this dataset")
    
    return {
        "message": f"Download started for {dataset_name}",
        "dataset": dataset_name
    }

@router.get("/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """Get detailed information about a specific dataset"""
    info = dataset_manager.get_dataset_info(dataset_name)
    if not info:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return info

@router.get("/{dataset_name}/sample")
async def get_dataset_sample(dataset_name: str, limit: int = 10):
    """Get a sample of records from a dataset"""
    try:
        if dataset_name == "liar":
            df = dataset_manager.load_liar_dataset()
        elif dataset_name == "politifact":
            df = dataset_manager.load_politifact_dataset()
        else:
            raise HTTPException(status_code=400, detail="Dataset loading not implemented")
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found or not downloaded")
        
        sample = df.head(limit).to_dict('records')
        return {
            "dataset": dataset_name,
            "total_records": len(df),
            "sample_size": len(sample),
            "sample": sample
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

@router.post("/{dataset_name}/preprocess", response_model=PreprocessingResponse)
async def preprocess_dataset(
    dataset_name: str, 
    request: PreprocessingRequest,
    background_tasks: BackgroundTasks
):
    """Preprocess a dataset with the full pipeline"""
    try:
        # Load the dataset
        if dataset_name == "liar":
            df = dataset_manager.load_liar_dataset()
        elif dataset_name == "politifact":
            df = dataset_manager.load_politifact_dataset()
        else:
            raise HTTPException(status_code=400, detail="Dataset loading not implemented")
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found or not downloaded")
        
        # Check if required columns exist
        if request.text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Text column '{request.text_column}' not found")
        
        if request.label_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Label column '{request.label_column}' not found")
        
        # Start preprocessing in background
        background_tasks.add_task(
            _preprocess_dataset_task,
            df,
            request.text_column,
            request.label_column,
            request.balance_strategy,
            request.save_path or f"processed_data/{dataset_name}_processed.csv"
        )
        
        return PreprocessingResponse(
            success=True,
            message=f"Preprocessing started for {dataset_name}",
            processed_records=len(df)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting preprocessing: {str(e)}")

@router.post("/preprocess/text")
async def preprocess_single_text(text: str):
    """Preprocess a single text string"""
    try:
        processed_text = preprocessor.preprocess_text(text)
        readability_features = preprocessor.calculate_readability_features(text)
        sentiment_features = preprocessor.calculate_sentiment_features(text)
        linguistic_features = preprocessor.calculate_linguistic_features(text)
        
        return {
            "original_text": text,
            "processed_text": processed_text,
            "features": {
                "readability": readability_features,
                "sentiment": sentiment_features,
                "linguistic": linguistic_features
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing text: {str(e)}")

@router.get("/preprocessing/status")
async def get_preprocessing_status():
    """Get status of preprocessing tasks"""
    # This would need to be implemented with a task queue like Celery
    # For now, just return a placeholder
    return {
        "message": "Preprocessing status tracking not implemented yet",
        "suggestion": "Use a task queue like Celery for production"
    }

async def _preprocess_dataset_task(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    balance_strategy: Optional[str],
    save_path: str
):
    """Background task for dataset preprocessing"""
    try:
        # Preprocess the dataset
        processed_df = preprocessor.preprocess_dataframe(df, text_column, label_column)
        
        # Balance the dataset if requested
        if balance_strategy:
            processed_df = preprocessor.balance_dataset(processed_df, 'label_encoded', balance_strategy)
        
        # Save the processed data
        preprocessor.save_preprocessed_data(processed_df, save_path)
        
        print(f"Preprocessing completed. Saved to {save_path}")
        
    except Exception as e:
        print(f"Error in preprocessing task: {e}")

@router.post("/full-pipeline/{dataset_name}")
async def full_pipeline_processing(
    dataset_name: str,
    request: FullPipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Complete pipeline: Download dataset (if needed), preprocess, and return markdown report
    
    This endpoint handles the entire workflow:
    1. Downloads the dataset if it doesn't exist or if download_if_missing=True
    2. Preprocesses the dataset with all features
    3. Applies balancing if requested
    4. Generates a comprehensive markdown report
    5. Returns the markdown file for download
    """
    try:
        # Step 1: Check if dataset exists and download if needed
        download_results = {}
        if request.download_if_missing:
            # Check if dataset exists
            dataset_info = dataset_manager.get_dataset_info(dataset_name)
            if not dataset_info or dataset_info.get('status') != 'downloaded':
                # Download the dataset
                if dataset_name == "liar":
                    success = dataset_manager.download_liar_dataset()
                elif dataset_name == "politifact":
                    success = dataset_manager.fetch_politifact_data()
                elif dataset_name == "fakenewsnet":
                    success = dataset_manager.download_fakenewsnet_dataset()
                else:
                    raise HTTPException(status_code=400, detail=f"Download not implemented for {dataset_name}")
                
                download_results[dataset_name] = success
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to download {dataset_name} dataset")
            else:
                download_results[dataset_name] = True
        else:
            download_results[dataset_name] = True
        
        # Step 2: Load the dataset
        if dataset_name == "liar":
            df = dataset_manager.load_liar_dataset()
        elif dataset_name == "politifact":
            df = dataset_manager.load_politifact_dataset()
        else:
            raise HTTPException(status_code=400, detail=f"Dataset loading not implemented for {dataset_name}")
        
        if df is None:
            raise HTTPException(status_code=404, detail=f"Could not load {dataset_name} dataset")
        
        # Step 3: Check required columns
        if request.text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Text column '{request.text_column}' not found")
        
        if request.label_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Label column '{request.label_column}' not found")
        
        # Step 4: Preprocess the dataset
        original_df = df.copy()
        processed_df = preprocessor.preprocess_dataframe(df, request.text_column, request.label_column)
        
        # Step 5: Apply balancing if requested
        balanced_df = None
        if request.balance_strategy:
            balanced_df = preprocessor.balance_dataset(processed_df, 'label_encoded', request.balance_strategy)
            final_df = balanced_df
        else:
            final_df = processed_df
        
        # Step 6: Save processed data
        output_dir = Path("processed_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_dir / f"{dataset_name}_processed.csv"
        preprocessor.save_preprocessed_data(final_df, str(csv_file))
        
        # Step 7: Generate markdown report
        if request.return_markdown:
            report_file = generate_markdown_report_api(
                results=download_results,
                original_df=original_df,
                processed_df=processed_df,
                balanced_df=balanced_df,
                dataset_name=dataset_name,
                output_dir=output_dir,
                report_type="full_pipeline"
            )
            
            # Return the markdown file for download
            return FileResponse(
                path=report_file,
                media_type='text/markdown',
                filename=f"{dataset_name}_full_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
        else:
            # Return JSON response with summary
            return {
                "success": True,
                "message": f"Full pipeline completed for {dataset_name}",
                "download_results": download_results,
                "original_records": len(original_df),
                "processed_records": len(processed_df),
                "final_records": len(final_df),
                "features_added": len(processed_df.columns) - len(original_df.columns),
                "balance_strategy": request.balance_strategy,
                "files_generated": {
                    "processed_csv": str(csv_file),
                    "label_mapping": str(output_dir / f"{dataset_name}_processed_label_mapping.json")
                }
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in full pipeline: {str(e)}")

@router.post("/full-pipeline/{dataset_name}/background")
async def full_pipeline_processing_background(
    dataset_name: str,
    request: FullPipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Complete pipeline in background: Download dataset (if needed), preprocess, and save markdown report
    
    This endpoint runs the full pipeline in the background and returns immediately.
    The markdown report will be saved to the processed_data directory.
    """
    try:
        # Add the full pipeline task to background tasks
        background_tasks.add_task(
            _full_pipeline_background_task,
            dataset_name,
            request
        )
        
        return {
            "success": True,
            "message": f"Full pipeline started for {dataset_name} in background",
            "dataset": dataset_name,
            "estimated_time": "2-5 minutes depending on dataset size"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting background pipeline: {str(e)}")

async def _full_pipeline_background_task(dataset_name: str, request: FullPipelineRequest):
    """Background task for full pipeline processing"""
    try:
        # Step 1: Download if needed
        if request.download_if_missing:
            dataset_info = dataset_manager.get_dataset_info(dataset_name)
            if not dataset_info or dataset_info.get('status') != 'downloaded':
                if dataset_name == "liar":
                    dataset_manager.download_liar_dataset()
                elif dataset_name == "politifact":
                    dataset_manager.fetch_politifact_data()
                elif dataset_name == "fakenewsnet":
                    dataset_manager.download_fakenewsnet_dataset()
        
        # Step 2: Load and process
        if dataset_name == "liar":
            df = dataset_manager.load_liar_dataset()
        elif dataset_name == "politifact":
            df = dataset_manager.load_politifact_dataset()
        
        if df is None:
            print(f"Error: Could not load {dataset_name} dataset")
            return
        
        # Step 3: Preprocess
        original_df = df.copy()
        processed_df = preprocessor.preprocess_dataframe(df, request.text_column, request.label_column)
        
        # Step 4: Balance if requested
        balanced_df = None
        if request.balance_strategy:
            balanced_df = preprocessor.balance_dataset(processed_df, 'label_encoded', request.balance_strategy)
            final_df = balanced_df
        else:
            final_df = processed_df
        
        # Step 5: Save data
        output_dir = Path("processed_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_dir / f"{dataset_name}_processed.csv"
        preprocessor.save_preprocessed_data(final_df, str(csv_file))
        
        # Step 6: Generate markdown report
        if request.return_markdown:
            download_results = {dataset_name: True}
            report_file = generate_markdown_report_api(
                results=download_results,
                original_df=original_df,
                processed_df=processed_df,
                balanced_df=balanced_df,
                dataset_name=dataset_name,
                output_dir=output_dir,
                report_type="full_pipeline_background"
            )
            print(f"Background pipeline completed. Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error in background pipeline task: {e}") 