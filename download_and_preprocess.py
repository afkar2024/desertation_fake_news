#!/usr/bin/env python3
"""
CLI script to download datasets and run preprocessing
Based on the pilot report requirements
"""

import asyncio
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import os

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.dataset_manager import dataset_manager
from app.preprocessing import preprocessor

def generate_markdown_report(results: Optional[Dict[str, bool]] = None, 
                           original_df: Optional[pd.DataFrame] = None, 
                           processed_df: Optional[pd.DataFrame] = None, 
                           balanced_df: Optional[pd.DataFrame] = None, 
                           dataset_name: str = "", 
                           output_dir: Optional[Path] = None,
                           report_type: str = "combined") -> str:
    """Generate a comprehensive markdown report"""
    
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



async def main():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets for fake news detection")
    parser.add_argument("--download", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str, help="Download specific dataset (liar, politifact, fakenewsnet)")
    parser.add_argument("--preprocess", type=str, help="Preprocess specific dataset")
    parser.add_argument("--balance", type=str, choices=["undersample", "oversample"], help="Balance dataset")
    parser.add_argument("--output", type=str, help="Output directory for processed data")
    parser.add_argument("--report", action="store_true", help="Generate detailed reports")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output) if args.output else Path("processed_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.download:
        print("🔄 Downloading all datasets...")
        start_time = time.time()
        results = dataset_manager.download_all_datasets()
        end_time = time.time()
        
        print(f"✅ Download results: {results}")
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        
        if args.report:
            report_file = generate_markdown_report(results=results, output_dir=output_dir, report_type="download")
            print(f"📊 Download report saved to: {report_file}")
    
    elif args.dataset:
        print(f"🔄 Downloading {args.dataset} dataset...")
        start_time = time.time()
        
        if args.dataset == "liar":
            success = dataset_manager.download_liar_dataset()
        elif args.dataset == "politifact":
            success = dataset_manager.fetch_politifact_data()
        elif args.dataset == "fakenewsnet":
            success = dataset_manager.download_fakenewsnet_dataset()
        else:
            print(f"❌ Unknown dataset: {args.dataset}")
            return
        
        end_time = time.time()
        
        if success:
            print(f"✅ Successfully downloaded {args.dataset}")
            print(f"⏱️  Download time: {end_time - start_time:.2f} seconds")
        else:
            print(f"❌ Failed to download {args.dataset}")
        
        if args.report and success:
            results = {args.dataset: success}
            report_file = generate_markdown_report(results=results, output_dir=output_dir, report_type="download", dataset_name=args.dataset)
            print(f"📊 Download report saved to: {report_file}")
    
    elif args.preprocess:
        print(f"🔄 Preprocessing {args.preprocess} dataset...")
        start_time = time.time()
        
        # Load the dataset
        if args.preprocess == "liar":
            df = dataset_manager.load_liar_dataset()
            text_column = "statement"
            label_column = "label"
        elif args.preprocess == "politifact":
            df = dataset_manager.load_politifact_dataset()
            text_column = "statement"
            label_column = "label"
        else:
            print(f"❌ Preprocessing not implemented for {args.preprocess}")
            return
        
        if df is None:
            print(f"❌ Could not load {args.preprocess} dataset. Please download it first.")
            return
        
        print(f"📊 Loaded {len(df)} records from {args.preprocess}")
        original_df = df.copy()
        
        # Preprocess the dataset
        processed_df = preprocessor.preprocess_dataframe(df, text_column, label_column)
        
        # Balance if requested
        balanced_df = None
        if args.balance:
            balanced_df = preprocessor.balance_dataset(processed_df, 'label_encoded', args.balance)
            final_df = balanced_df
        else:
            final_df = processed_df
        
        # Save processed data
        output_file = output_dir / f"{args.preprocess}_processed.csv"
        preprocessor.save_preprocessed_data(final_df, str(output_file))
        
        end_time = time.time()
        
        print(f"✅ Preprocessing completed!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Final dataset shape: {final_df.shape}")
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
        
        # Show feature summary
        if hasattr(final_df, 'attrs') and 'label_mapping' in final_df.attrs:
            print(f"🏷️  Label mapping: {final_df.attrs['label_mapping']}")
        
        # Generate detailed report
        if args.report:
            report_file = generate_markdown_report(
                original_df=original_df, 
                processed_df=processed_df, 
                balanced_df=balanced_df, 
                dataset_name=args.preprocess, 
                output_dir=output_dir,
                report_type="preprocessing"
            )
            print(f"📊 Detailed preprocessing report saved to: {report_file}")
    
    else:
        print("❓ Please specify an action: --download, --dataset, or --preprocess")
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main()) 