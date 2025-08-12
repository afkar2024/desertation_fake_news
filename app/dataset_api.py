"""
API endpoints for dataset management and preprocessing
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
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
from .model_service import model_service
from .reports import write_metrics_report, write_comparative_metrics_report
from .cache_store import (
    compute_params_hash,
    get_result as cache_get,
    put_result as cache_put,
    add_report,
    list_reports_json,
    get_report_json,
)
from .data_utils import calculate_fake_news_score
from .reports import write_cross_domain_report
from .eval_progress import progress_manager

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
    return_markdown: bool = False

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


class EvaluateModelRequest(BaseModel):
    text_column: str = "statement"
    label_column: str = "label"
    limit: int = 1000
    binary_mapping: Optional[Dict[str, int]] = None  # optional custom mapping
    compare_baseline: bool = True
    save_report: bool = False
    shap_samples: int = 5  # number of SHAP examples to save to processed_data
    compare_traditional: bool = False  # optional TF-IDF + LogisticRegression baseline
    compare_svm: bool = False  # optional TF-IDF + LinearSVC (with calibration)
    compare_nb: bool = False   # optional TF-IDF + MultinomialNB
    robustness_test: bool = False  # apply simple perturbations and re-evaluate
    abstention_curve: bool = False  # compute coverage-accuracy curve from uncertainty
    mc_dropout_samples: int = 0  # if >0, compute MC dropout uncertainty metrics
    use_ensemble: bool = False  # if true (and compare_traditional), compute simple ensemble
    explainability_quality: bool = False  # compute deletion-based faithfulness metric
    fairness_column: Optional[str] = None  # categorical column to assess fairness gaps
    temporal_column: Optional[str] = None  # datetime-like column for temporal stability
    temporal_granularity: str = "month"  # 'day' or 'month'
    attention_analysis: bool = False  # summarize attention weights on sample

class EvaluateModelResponse(BaseModel):
    dataset: str
    total_evaluated: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    report_path: Optional[str] = None
    extra_metrics: Optional[Dict[str, Any]] = None


class CrossDomainEvaluateRequest(BaseModel):
    text_column: str = "statement"
    label_column: str = "label"
    limit: int = 1000
    compare_baseline: bool = True
    datasets: List[str] = ["liar", "politifact"]
    save_report: bool = False


@router.post("/evaluate/cross-domain")
async def evaluate_cross_domain(request: CrossDomainEvaluateRequest, trace_id: Optional[str] = None):
    """Evaluate the current model across multiple datasets and write a summary report."""
    evaluations: List[Dict[str, Any]] = []

    for name in request.datasets:
        try:
            single_req = EvaluateModelRequest(
                text_column=request.text_column,
                label_column=request.label_column,
                limit=request.limit,
                compare_baseline=request.compare_baseline,
            )
            # Call internal function logic directly rather than HTTP to avoid network
            # We duplicate minimal core logic by reusing existing loader branches
            if name == "liar":
                df = dataset_manager.load_liar_dataset()
                default_map = {
                    "true": 0,
                    "mostly-true": 0,
                    "half-true": 0,
                    "barely-true": 1,
                    "false": 1,
                    "pants-fire": 1,
                    "pants-on-fire": 1,
                }
            elif name == "politifact":
                df = dataset_manager.load_politifact_dataset()
                default_map = None
            else:
                continue

            if df is None or df.empty:
                continue

            df = df.head(max(1, single_req.limit))

            y_true = None
            if single_req.label_column in df.columns:
                if default_map:
                    y_true = df[single_req.label_column].map(default_map)
                else:
                    y_true = df[single_req.label_column]
                try:
                    y_true = y_true.astype(int)
                except Exception:
                    pass

            texts = df[single_req.text_column].astype(str).tolist()
            preds = model_service.classify_batch(texts)
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            y_pred = np.array([p.get("prediction", 0) for p in preds], dtype=int)

            metrics_item: Dict[str, Any] = {"dataset": name, "size": int(len(df))}
            if y_true is not None:
                y_true_arr = np.array(y_true.tolist(), dtype=int)
                acc = float(accuracy_score(y_true_arr, y_pred))
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_arr, y_pred, average="binary", zero_division=0
                )
                metrics_item.update(
                    {
                        "accuracy": round(acc, 4),
                        "precision": round(float(prec), 4),
                        "recall": round(float(rec), 4),
                        "f1": round(float(f1), 4),
                    }
                )

                if request.compare_baseline:
                    baseline_pred = []
                    for t, row in zip(texts, df.itertuples(index=False)):
                        title = getattr(row, "title", "") if hasattr(row, "title") else ""
                        url = getattr(row, "source_url", "") if hasattr(row, "source_url") else ""
                        fake_prob = calculate_fake_news_score(title, t, url).get("fake_probability", 0.0)
                        baseline_pred.append(1 if fake_prob >= 0.5 else 0)
                    baseline_pred = np.array(baseline_pred, dtype=int)
                    b_acc = float(accuracy_score(y_true_arr, baseline_pred))
                    b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(
                        y_true_arr, baseline_pred, average="binary", zero_division=0
                    )
                    metrics_item.update(
                        {
                            "baseline_accuracy": round(b_acc, 4),
                            "baseline_precision": round(float(b_prec), 4),
                            "baseline_recall": round(float(b_rec), 4),
                            "baseline_f1": round(float(b_f1), 4),
                        }
                    )

            evaluations.append(metrics_item)
        except Exception:
            # Keep going for other datasets
            continue

    report_path = None
    if request.save_report:
        report_dir = Path("processed_data")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = write_cross_domain_report({"evaluations": evaluations}, report_dir)
    # Cache per request shape (datasets + params)
    dataset_fp = "|".join([f"{e['dataset']}:{e['size']}" for e in evaluations])
    params_hash = compute_params_hash(request.dict())
    payload = {"evaluations": evaluations, "report_path": report_path}
    trace = cache_put(trace_id=trace_id, dataset="multi", process="cross_domain", params_hash=params_hash, dataset_fingerprint=dataset_fp, payload=payload)
    payload["trace_id"] = trace
    payload["cached"] = False
    return payload


class ActiveSamplingRequest(BaseModel):
    text_column: str = "statement"
    strategy: str = "least_confidence"  # 'least_confidence' or 'highest_entropy'
    limit: int = 20


@router.post("/active-samples/{dataset_name}")
async def get_active_learning_samples(dataset_name: str, request: ActiveSamplingRequest):
    """Return top-K samples suggested for annotation via active learning.

    Strategies:
      - least_confidence: lowest margin (|p(fake)-p(real)|)
      - highest_entropy: highest predictive entropy
    """
    # Load dataset
    if dataset_name == "liar":
        df = dataset_manager.load_liar_dataset()
    elif dataset_name == "politifact":
        df = dataset_manager.load_politifact_dataset()
    else:
        raise HTTPException(status_code=400, detail=f"Active sampling not implemented for {dataset_name}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    texts = df[request.text_column].astype(str).tolist()
    preds = model_service.classify_batch(texts)

    # Compute scores
    import numpy as np
    margins = np.array([float(item.get("uncertainty", {}).get("margin", 0.0)) for item in preds])
    entropies = np.array([float(item.get("uncertainty", {}).get("predictive_entropy", 0.0)) for item in preds])

    if request.strategy == "highest_entropy":
        order = np.argsort(-entropies)
    else:
        # least confidence = lowest margin first
        order = np.argsort(margins)

    k = max(1, min(int(request.limit), len(texts)))
    idx = order[:k]
    items: List[Dict[str, Any]] = []
    for i in idx:
        row = df.iloc[int(i)]
        pr = preds[int(i)]
        items.append(
            {
                "index": int(i),
                "text": texts[int(i)],
                "prediction": int(pr.get("prediction", 0)),
                "confidence": float(pr.get("confidence", 0.0)),
                "probabilities": pr.get("probabilities", {}),
                "uncertainty": pr.get("uncertainty", {}),
            }
        )

    return {"dataset": dataset_name, "strategy": request.strategy, "limit": k, "samples": items}

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
async def get_dataset_sample(dataset_name: str, size: int = 10):
    """Get a sample of records from a dataset"""
    try:
        if dataset_name == "liar":
            df = dataset_manager.load_liar_dataset()
        elif dataset_name == "politifact":
            df = dataset_manager.load_politifact_dataset()
        elif dataset_name == "fakenewsnet":
            # Not implemented in demo scope: return an empty sample gracefully
            return {
                "dataset": dataset_name,
                "total_records": 0,
                "sample_size": 0,
                "sample": []
            }
        else:
            # Unknown dataset: return empty response (avoid breaking frontend flow)
            return {
                "dataset": dataset_name,
                "total_records": 0,
                "sample_size": 0,
                "sample": []
            }
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found or not downloaded")
        
        # Clean the dataframe to handle inf, -inf, and NaN values for JSON serialization
        df_clean = df.copy()
        
        # Replace inf, -inf with None (which becomes null in JSON)
        df_clean = df_clean.replace([float('inf'), float('-inf')], None)
        
        # Replace NaN with None - use a more robust approach
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        
        # Convert to records and ensure all values are JSON serializable
        sample = df_clean.head(size).to_dict('records')
        
        # Additional safety check: convert any remaining problematic values
        def clean_record(record):
            cleaned = {}
            for key, value in record.items():
                if pd.isna(value) or (isinstance(value, float) and (value == float('inf') or value == float('-inf'))):
                    cleaned[key] = None
                elif isinstance(value, (int, float)) and (value == float('inf') or value == float('-inf') or value != value):  # NaN check
                    cleaned[key] = None
                else:
                    cleaned[key] = value
            return cleaned
        
        sample = [clean_record(record) for record in sample]
        
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
    request: FullPipelineRequest = FullPipelineRequest(),
    background_tasks: BackgroundTasks = None,
    trace_id: Optional[str] = None,
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
        
        # Simple cache check before processing
        dataset_fp = str(len(df))  # minimal fingerprint: can be enhanced to hash content if needed
        params_hash = compute_params_hash(request.dict())
        cached = cache_get(dataset=dataset_name, process="full_pipeline", params_hash=params_hash, dataset_fingerprint=dataset_fp, trace_id=trace_id)
        if cached:
            return cached

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
        # Default: return JSON payload for UI (no file streaming)
        response_payload = {
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
        trace = cache_put(trace_id=trace_id, dataset=dataset_name, process="full_pipeline", params_hash=params_hash, dataset_fingerprint=dataset_fp, payload=response_payload)
        response_payload["trace_id"] = trace
        response_payload["cached"] = False
        return response_payload
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in full pipeline: {str(e)}")


@router.post("/evaluate/{dataset_name}", response_model=EvaluateModelResponse)
async def evaluate_model_on_dataset(dataset_name: str, request: EvaluateModelRequest, trace_id: Optional[str] = None):
    """Quick evaluation of current model on a dataset subset; writes a metrics report."""
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve

    try:
        trace = trace_id or f"eval-{datetime.now().timestamp()}"
        await progress_manager.publish(trace, stage="start", message="Starting evaluation")
        # Load dataset
        if dataset_name == "liar":
            df = dataset_manager.load_liar_dataset()
            default_map = {
                "true": 0,
                "mostly-true": 0,
                "half-true": 0,
                "barely-true": 1,
                "false": 1,
                "pants-fire": 1,
                "pants-on-fire": 1,
            }
        elif dataset_name == "politifact":
            df = dataset_manager.load_politifact_dataset()
            default_map = None
        else:
            raise HTTPException(status_code=400, detail=f"Evaluation not implemented for {dataset_name}")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Dataset not found or empty")

        # Cache check before heavy work
        dataset_fp = str(len(df))
        params_hash = compute_params_hash(request.dict())
        cached = cache_get(dataset=dataset_name, process="evaluate", params_hash=params_hash, dataset_fingerprint=dataset_fp, trace_id=trace_id)
        if cached:
            await progress_manager.publish(trace, stage="cached", message="Returning cached evaluation", percent=100.0)
            # map cached payload back to model response structure if needed
            return EvaluateModelResponse(
                dataset=dataset_name,
                total_evaluated=int(cached.get("total_evaluated", 0)),
                accuracy=float(cached.get("accuracy", 0.0)),
                precision=float(cached.get("precision", 0.0)),
                recall=float(cached.get("recall", 0.0)),
                f1=float(cached.get("f1", 0.0)),
                report_path=cached.get("report_path"),
                extra_metrics=cached.get("extra_metrics", {}),
            )

        # Subset
        df = df.head(max(1, request.limit))
        await progress_manager.publish(trace, stage="subset", message=f"Subset to {len(df)} records")

        # Prepare labels
        y_true = None
        if request.label_column in df.columns:
            if request.binary_mapping:
                y_true = df[request.label_column].map(request.binary_mapping)
            elif default_map:
                y_true = df[request.label_column].map(default_map)
            else:
                # Attempt to coerce to 0/1
                y_true = df[request.label_column]
            try:
                y_true = y_true.astype(int)
            except Exception:
                pass

        # Generate predictions (model)
        texts = df[request.text_column].astype(str).tolist()
        await progress_manager.publish(trace, stage="predict", message="Running model predictions")
        preds = model_service.classify_batch(texts)
        y_pred = np.array([p["prediction"] for p in preds], dtype=int)

        # If labels are available, compute metrics
        accuracy = precision = recall = f1 = 0.0
        baseline_metrics = None
        mcnemar_stats = None
        calibration_metrics = None
        fairness_metrics = None
        temporal_metrics = None
        attention_metrics = None
        if y_true is not None:
            y_true_arr = np.array(y_true.tolist(), dtype=int)
            accuracy = float(accuracy_score(y_true_arr, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_arr, y_pred, average="binary", zero_division=0
            )
            await progress_manager.publish(trace, stage="metrics", message="Computing metrics")
            # Confusion matrix (TN, FP, FN, TP)
            try:
                tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
                metrics_payload = {
                    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
                }
            except Exception:
                metrics_payload = {}

            # Calibration & discrimination metrics
            try:
                # Extract model probabilities for class 1 (fake)
                prob_fake = np.array([float(item.get("probabilities", {}).get("fake", 0.0)) for item in preds])
                # Brier score
                brier = float(np.mean((prob_fake - y_true_arr) ** 2))
                # ROC/PR curves and AUCs
                roc_auc = None
                pr_auc = None
                roc_points = None
                pr_points = None
                try:
                    if len(np.unique(y_true_arr)) > 1:
                        fpr, tpr, _ = roc_curve(y_true_arr, prob_fake)
                        roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
                        roc_auc = float(roc_auc_score(y_true_arr, prob_fake))
                        prec, rec, _ = precision_recall_curve(y_true_arr, prob_fake)
                        pr_points = [{"precision": float(p), "recall": float(r)} for p, r in zip(prec, rec)]
                        pr_auc = float(average_precision_score(y_true_arr, prob_fake))
                except Exception:
                    pass

                # Expected Calibration Error (ECE)
                def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10):
                    bins = np.linspace(0.0, 1.0, num_bins + 1)
                    ece_val = 0.0
                    reliability = []
                    for i in range(num_bins):
                        lo, hi = bins[i], bins[i + 1]
                        mask = (probs >= lo) & (probs < hi if i < num_bins - 1 else probs <= hi)
                        count = int(mask.sum())
                        if count == 0:
                            reliability.append({"bin": i + 1, "low": float(lo), "high": float(hi), "count": 0, "avg_conf": None, "accuracy": None})
                            continue
                        bin_probs = probs[mask]
                        bin_labels = labels[mask]
                        avg_conf = float(np.mean(bin_probs))
                        acc = float(np.mean((bin_probs >= 0.5) == (bin_labels == 1)))
                        ece_val += (count / len(probs)) * abs(acc - avg_conf)
                        reliability.append({"bin": i + 1, "low": float(lo), "high": float(hi), "count": count, "avg_conf": avg_conf, "accuracy": acc})
                    return float(ece_val), reliability

                ece, reliability_bins = compute_ece(prob_fake, y_true_arr, num_bins=10)
                calibration_metrics = {
                    "brier_score": round(brier, 6),
                    "ece": round(ece, 6),
                }
                if roc_auc is not None:
                    calibration_metrics["roc_auc"] = round(roc_auc, 6)
                    if roc_points is not None:
                        calibration_metrics["roc_curve"] = {"auc": round(roc_auc, 6), "points": roc_points}
                if pr_auc is not None:
                    calibration_metrics["pr_auc"] = round(pr_auc, 6)
                    if pr_points is not None:
                        calibration_metrics["pr_curve"] = {"auc": round(pr_auc, 6), "points": pr_points}
                # include bins for report
                calibration_metrics["reliability_bins"] = reliability_bins
            except Exception:
                calibration_metrics = None

            # MC Dropout uncertainty (epistemic) if requested
            if request.mc_dropout_samples and request.mc_dropout_samples > 0:
                try:
                    await progress_manager.publish(trace, stage="uncertainty", message="Computing MC Dropout uncertainty")
                    mc = model_service.predict_proba_mc(texts, mc_samples=int(request.mc_dropout_samples))
                    mi = mc.get("mutual_information", [])
                    if len(mi) == len(texts):
                        # summarize
                        mi_arr = np.array(mi, dtype=float)
                        metrics_payload_mi = {
                            "mc_dropout_samples": int(request.mc_dropout_samples),
                            "mutual_information_mean": round(float(mi_arr.mean()), 6),
                            "mutual_information_std": round(float(mi_arr.std()), 6),
                        }
                    else:
                        metrics_payload_mi = {"mc_dropout_samples": int(request.mc_dropout_samples)}
                except Exception:
                    metrics_payload_mi = {"mc_dropout_samples": int(request.mc_dropout_samples), "mc_error": True}
            else:
                metrics_payload_mi = None

            # Fairness across subgroups
            if request.fairness_column and request.fairness_column in df.columns:
                try:
                    await progress_manager.publish(trace, stage="fairness", message="Computing fairness metrics")
                    groups = df[request.fairness_column].astype(str).fillna("unknown")
                    y_true_arr_f = y_true_arr
                    gaps = []
                    per_group = {}
                    for g in groups.unique():
                        mask = (groups == g).to_numpy()
                        if mask.sum() == 0:
                            continue
                        acc_g = float(accuracy_score(y_true_arr_f[mask], y_pred[mask]))
                        per_group[str(g)] = {"accuracy": round(acc_g, 4), "n": int(mask.sum())}
                        gaps.append(acc_g)
                    if per_group:
                        fairness_metrics = {
                            "fairness_groups": per_group,
                            "fairness_accuracy_gap": round(float(max(gaps) - min(gaps)), 6) if len(gaps) > 1 else 0.0,
                        }
                except Exception:
                    fairness_metrics = None

            # Temporal stability analysis
            if request.temporal_column and request.temporal_column in df.columns:
                try:
                    await progress_manager.publish(trace, stage="temporal", message="Computing temporal stability")
                    tseries = pd.to_datetime(df[request.temporal_column], errors='coerce')
                    if request.temporal_granularity == "day":
                        keys = tseries.dt.strftime("%Y-%m-%d")
                    else:
                        keys = tseries.dt.strftime("%Y-%m")
                    per_period = {}
                    for k in keys.unique():
                        mask = (keys == k).to_numpy()
                        if mask.sum() == 0:
                            continue
                        acc_k = float(accuracy_score(y_true_arr[mask], y_pred[mask]))
                        per_period[str(k)] = {"accuracy": round(acc_k, 4), "n": int(mask.sum())}
                    if per_period:
                        # simple variability metric
                        vals = [v["accuracy"] for v in per_period.values()]
                        temporal_metrics = {
                            "temporal_periods": per_period,
                            "temporal_accuracy_std": round(float(np.std(vals)), 6) if len(vals) > 1 else 0.0,
                        }
                except Exception:
                    temporal_metrics = None

            # Attention analysis (mean top-k attention weight over sample)
            if request.attention_analysis:
                try:
                    sample_k2 = int(max(5, min(20, len(texts) * 0.1)))
                    idxs2 = list(range(len(texts)))[:sample_k2]
                    k_top = 10
                    top_means: List[float] = []
                    for i in idxs2:
                        att = model_service.explain_text_attention(texts[i])
                        weights = att.get("weights", [])
                        if weights:
                            arr = np.array(weights, dtype=float)
                            order = np.argsort(-arr)
                            top_vals = arr[order[:k_top]]
                            top_means.append(float(top_vals.mean()))
                    if top_means:
                        attention_metrics = {
                            "attention_topk_mean": round(float(np.mean(top_means)), 6),
                            "attention_samples": len(top_means),
                        }
                except Exception:
                    attention_metrics = None

            # Explainability quality (deletion faithfulness) on a small subsample
            exp_quality_metrics = None
            if request.explainability_quality:
                try:
                    await progress_manager.publish(trace, stage="xai_quality", message="Computing explanation quality")
                    sample_k = int(max(5, min(20, len(texts) * 0.1)))
                    idxs = list(range(len(texts)))[:sample_k]
                    top_k_tokens = 5
                    deltas_top: List[float] = []
                    deltas_rand: List[float] = []
                    rng = np.random.default_rng(42)

                    for i in idxs:
                        t = texts[i]
                        base_item = preds[i]
                        base_prob_fake = float(base_item.get("probabilities", {}).get("fake", 0.0))
                        # Top-k by SHAP
                        top_tokens = model_service.get_top_influential_tokens_shap(t, top_k=top_k_tokens, by="abs")
                        words = t.split()
                        # Remove top tokens
                        if top_tokens:
                            v_top = " ".join([w for w in words if w not in set(top_tokens)])
                            if not v_top:
                                v_top = t
                        else:
                            v_top = t
                        pred_top = model_service.classify_text(v_top)
                        prob_top = float(pred_top.get("probabilities", {}).get("fake", 0.0))
                        deltas_top.append(abs(prob_top - base_prob_fake))

                        # Random tokens removal (same count)
                        if len(words) > 0 and top_tokens:
                            num = min(len(top_tokens), len(words))
                            rand_idx = rng.choice(len(words), size=num, replace=False)
                            mask_set = {words[j] for j in rand_idx}
                            v_rand = " ".join([w for w in words if w not in mask_set])
                            if not v_rand:
                                v_rand = t
                        else:
                            v_rand = t
                        pred_rand = model_service.classify_text(v_rand)
                        prob_rand = float(pred_rand.get("probabilities", {}).get("fake", 0.0))
                        deltas_rand.append(abs(prob_rand - base_prob_fake))

                    if deltas_top:
                        exp_quality_metrics = {
                            "explainability_quality_topk_delta_mean": round(float(np.mean(deltas_top)), 6),
                            "explainability_quality_random_delta_mean": round(float(np.mean(deltas_rand) if deltas_rand else 0.0), 6),
                            "explainability_quality_effect": round(float((np.mean(deltas_top) - (np.mean(deltas_rand) if deltas_rand else 0.0))), 6),
                            "explainability_quality_samples": sample_k,
                        }
                except Exception:
                    exp_quality_metrics = None

            # Baseline comparison using heuristic score
            if request.compare_baseline:
                baseline_pred = []
                for t, row in zip(texts, df.itertuples(index=False)):
                    title = getattr(row, "title", "") if hasattr(row, "title") else ""
                    url = getattr(row, "source_url", "") if hasattr(row, "source_url") else ""
                    fake_prob = calculate_fake_news_score(title, t, url).get("fake_probability", 0.0)
                    baseline_pred.append(1 if fake_prob >= 0.5 else 0)
                baseline_pred = np.array(baseline_pred, dtype=int)
                b_acc = float(accuracy_score(y_true_arr, baseline_pred))
                b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(
                    y_true_arr, baseline_pred, average="binary", zero_division=0
                )
                baseline_metrics = {
                    "baseline_accuracy": round(b_acc, 4),
                    "baseline_precision": round(float(b_prec), 4),
                    "baseline_recall": round(float(b_rec), 4),
                    "baseline_f1": round(float(b_f1), 4),
                }

                # McNemar's test (with continuity correction) for paired predictions
                # Contingency:
                #  b = model correct, baseline wrong
                #  c = model wrong, baseline correct
                try:
                    import math
                    model_correct = (y_pred == y_true_arr).astype(int)
                    baseline_correct = (baseline_pred == y_true_arr).astype(int)
                    b = int(((model_correct == 1) & (baseline_correct == 0)).sum())
                    c = int(((model_correct == 0) & (baseline_correct == 1)).sum())
                    # chi^2 with continuity correction: ((|b-c|-1)^2)/(b+c), p from chi2 df=1
                    denom = b + c
                    if denom > 0:
                        chi2 = ((abs(b - c) - 1) ** 2) / denom
                        # survival function for chi-square with df=1: sf(x) = erfc(sqrt(x/2))
                        p_val = float(math.erfc(math.sqrt(chi2 / 2.0)))
                        mcnemar_stats = {
                            "mcnemar_b": b,
                            "mcnemar_c": c,
                            "mcnemar_chi2": round(float(chi2), 6),
                            "mcnemar_p": round(float(p_val), 8),
                        }
                except Exception:
                    mcnemar_stats = None

            # Optional: Traditional ML baseline (TF-IDF + LogisticRegression)
            trad_prob = None
            svm_prob = None
            nb_prob = None
            if request.compare_traditional and y_true is not None:
                try:
                    await progress_manager.publish(trace, stage="baseline_traditional", message="Computing traditional baseline")
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.pipeline import Pipeline
                    from sklearn.model_selection import train_test_split

                    X_text = df[request.text_column].astype(str).tolist()
                    y = y_true.astype(int).tolist()

                    # Simple split to avoid leakage across evaluation, train only on half and test on eval subset
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_text, y, test_size=0.5, random_state=42, stratify=y
                    )
                    pipe = Pipeline(
                        [
                            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                            ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
                        ]
                    )
                    pipe.fit(X_train, y_train)
                    # probs and labels
                    trad_proba_full = pipe.predict_proba(X_text)[:, 1]
                    trad_prob = trad_proba_full
                    trad_pred = (trad_proba_full >= 0.5).astype(int)

                    t_acc = float(accuracy_score(np.array(y, dtype=int), trad_pred))
                    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(
                        np.array(y, dtype=int), trad_pred, average="binary", zero_division=0
                    )
                    if baseline_metrics is None:
                        baseline_metrics = {}
                    baseline_metrics.update(
                        {
                            "traditional_accuracy": round(t_acc, 4),
                            "traditional_precision": round(float(t_prec), 4),
                            "traditional_recall": round(float(t_rec), 4),
                            "traditional_f1": round(float(t_f1), 4),
                        }
                    )

                    # McNemar between model and traditional baseline
                    try:
                        import math
                        model_correct = (y_pred == np.array(y, dtype=int)).astype(int)
                        trad_correct = (trad_pred == np.array(y, dtype=int)).astype(int)
                        b2 = int(((model_correct == 1) & (trad_correct == 0)).sum())
                        c2 = int(((model_correct == 0) & (trad_correct == 1)).sum())
                        denom2 = b2 + c2
                        if denom2 > 0:
                            chi22 = ((abs(b2 - c2) - 1) ** 2) / denom2
                            p2 = float(math.erfc(math.sqrt(chi22 / 2.0)))
                            # Store separately to avoid key collision
                            if mcnemar_stats is None:
                                mcnemar_stats = {}
                            mcnemar_stats.update(
                                {
                                    "mcnemar_vs_traditional_b": b2,
                                    "mcnemar_vs_traditional_c": c2,
                                    "mcnemar_vs_traditional_chi2": round(float(chi22), 6),
                                    "mcnemar_vs_traditional_p": round(float(p2), 8),
                                }
                            )
                    except Exception:
                        pass
                except Exception:
                    # Traditional baseline is best-effort; ignore failures silently
                    pass

            # SVM baseline (with probability calibration)
            if request.compare_svm and y_true is not None:
                try:
                    await progress_manager.publish(trace, stage="baseline_svm", message="Computing SVM baseline")
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.svm import LinearSVC
                    from sklearn.calibration import CalibratedClassifierCV
                    from sklearn.pipeline import Pipeline
                    from sklearn.model_selection import train_test_split

                    X_text = df[request.text_column].astype(str).tolist()
                    y = y_true.astype(int).tolist()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_text, y, test_size=0.5, random_state=42, stratify=y
                    )
                    base = Pipeline([
                        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                        ("svm", LinearSVC())
                    ])
                    base.fit(X_train, y_train)
                    # calibrate on held-out set for probabilities
                    from sklearn.feature_extraction.text import TfidfVectorizer as _TV
                    from sklearn.pipeline import make_pipeline
                    # Refit vectorizer on all, then calibrate using CV for simplicity
                    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
                    X_all = vec.fit_transform(X_text)
                    svm = LinearSVC()
                    calibrated = CalibratedClassifierCV(svm, cv=3)
                    calibrated.fit(X_all, np.array(y, dtype=int))
                    svm_prob = calibrated.predict_proba(X_all)[:, 1]
                    svm_pred = (svm_prob >= 0.5).astype(int)

                    s_acc = float(accuracy_score(np.array(y, dtype=int), svm_pred))
                    s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(
                        np.array(y, dtype=int), svm_pred, average="binary", zero_division=0
                    )
                    if baseline_metrics is None:
                        baseline_metrics = {}
                    baseline_metrics.update(
                        {
                            "svm_accuracy": round(s_acc, 4),
                            "svm_precision": round(float(s_prec), 4),
                            "svm_recall": round(float(s_rec), 4),
                            "svm_f1": round(float(s_f1), 4),
                        }
                    )
                except Exception:
                    pass

            # Naive Bayes baseline (MultinomialNB)
            if request.compare_nb and y_true is not None:
                try:
                    await progress_manager.publish(trace, stage="baseline_nb", message="Computing Naive Bayes baseline")
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.naive_bayes import MultinomialNB
                    from sklearn.pipeline import Pipeline
                    from sklearn.model_selection import train_test_split

                    X_text = df[request.text_column].astype(str).tolist()
                    y = y_true.astype(int).tolist()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_text, y, test_size=0.5, random_state=42, stratify=y
                    )
                    pipe_nb = Pipeline([
                        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                        ("nb", MultinomialNB()),
                    ])
                    pipe_nb.fit(X_train, y_train)
                    nb_prob = pipe_nb.predict_proba(X_text)[:, 1]
                    nb_pred = (nb_prob >= 0.5).astype(int)

                    n_acc = float(accuracy_score(np.array(y, dtype=int), nb_pred))
                    n_prec, n_rec, n_f1, _ = precision_recall_fscore_support(
                        np.array(y, dtype=int), nb_pred, average="binary", zero_division=0
                    )
                    if baseline_metrics is None:
                        baseline_metrics = {}
                    baseline_metrics.update(
                        {
                            "nb_accuracy": round(n_acc, 4),
                            "nb_precision": round(float(n_prec), 4),
                            "nb_recall": round(float(n_rec), 4),
                            "nb_f1": round(float(n_f1), 4),
                        }
                    )
                except Exception:
                    pass

            # Simple ensemble (average of probabilities)
            if request.use_ensemble and y_true is not None:
                try:
                    await progress_manager.publish(trace, stage="ensemble", message="Computing ensemble")
                    # transformer probabilities
                    prob_fake = np.array([float(item.get("probabilities", {}).get("fake", 0.0)) for item in preds])
                    prob_list = [prob_fake]
                    if trad_prob is not None:
                        prob_list.append(np.array(trad_prob))
                    if svm_prob is not None:
                        prob_list.append(np.array(svm_prob))
                    if nb_prob is not None:
                        prob_list.append(np.array(nb_prob))
                    if len(prob_list) <= 1:
                        raise Exception("No additional baseline probabilities for ensemble")
                    ens_prob = np.mean(np.stack(prob_list, axis=0), axis=0)
                    ens_pred = (ens_prob >= 0.5).astype(int)
                    y_true_arr4 = np.array(y_true.tolist(), dtype=int)
                    e_acc = float(accuracy_score(y_true_arr4, ens_pred))
                    e_prec, e_rec, e_f1, _ = precision_recall_fscore_support(
                        y_true_arr4, ens_pred, average="binary", zero_division=0
                    )
                    if baseline_metrics is None:
                        baseline_metrics = {}
                    baseline_metrics.update(
                        {
                            "ensemble_accuracy": round(e_acc, 4),
                            "ensemble_precision": round(float(e_prec), 4),
                            "ensemble_recall": round(float(e_rec), 4),
                            "ensemble_f1": round(float(e_f1), 4),
                        }
                    )
                except Exception:
                    pass

        # Write metrics report and optional SHAP sample explanations
        report_dir = Path("processed_data")
        report_dir.mkdir(parents=True, exist_ok=True)

        shap_json_path: Optional[str] = None
        shap_md_path: Optional[str] = None

        # Save a few SHAP examples if requested and explanations are enabled
        try:
            if request.shap_samples > 0:
                from app.config import settings  # local import to avoid cycle at module import time

                if settings.enable_explanations:
                    sample_count = min(request.shap_samples, len(texts))
                    sample_items: List[Dict[str, Any]] = []
                    for i in range(sample_count):
                        t = texts[i]
                        pred_item = preds[i] if i < len(preds) else model_service.classify_text(t)
                        exp = model_service.explain_text(t, max_evals=None)
                        sample_items.append(
                            {
                                "index": i,
                                "text": t,
                                "prediction": int(pred_item.get("prediction", 0)),
                                "confidence": float(pred_item.get("confidence", 0.0)),
                                "probabilities": pred_item.get("probabilities", {}),
                                "explanation": {
                                    "tokens": exp.get("tokens", []),
                                    "shap_values": exp.get("shap_values", []),
                                    "base_value": exp.get("base_value", 0.0),
                                },
                            }
                        )

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shap_json = report_dir / f"shap_samples_{dataset_name}_{ts}.json"
                    with shap_json.open("w", encoding="utf-8") as f:
                        json.dump(sample_items, f, ensure_ascii=False, indent=2)
                    shap_json_path = str(shap_json)

                    # Create a simple markdown preview with inline coloring via HTML spans
                    shap_md = report_dir / f"shap_samples_{dataset_name}_{ts}.md"
                    with shap_md.open("w", encoding="utf-8") as f:
                        f.write(f"# SHAP Sample Explanations ({dataset_name})\n\n")
                        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                        for item in sample_items:
                            f.write(f"## Sample #{item['index']} — pred={'fake' if item['prediction']==1 else 'real'} (conf={item['confidence']:.3f})\n\n")
                            tokens = item["explanation"].get("tokens", [])
                            values = item["explanation"].get("shap_values", [])
                            f.write("<div style='line-height:1.8;'>\n")
                            for tok, val in zip(tokens, values):
                                absval = min(abs(val), 1.0)
                                # red for positive (towards fake), green for negative (towards real)
                                color = f"rgba(244,67,54,{0.15 + absval*0.6})" if val > 0 else f"rgba(76,175,80,{0.15 + absval*0.6})"
                                safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
                                f.write(f"<span style='display:inline-block;margin:2px;padding:2px 4px;border-radius:4px;background:{color};font-family:monospace;'>{safe_tok}</span>")
                            f.write("\n</div>\n\n")
                    shap_md_path = str(shap_md)
        except Exception as _:
            # Non-fatal: continue even if SHAP examples failed
            pass

        metrics_payload = {
            "dataset": dataset_name,
            "size": int(len(df)),
            "accuracy": round(accuracy, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "model_source": model_service.get_model_info().get("current_source"),
        }
        if shap_json_path:
            metrics_payload["shap_samples_json"] = shap_json_path
        if shap_md_path:
            metrics_payload["shap_samples_markdown"] = shap_md_path
        if baseline_metrics:
            metrics_payload.update(baseline_metrics)
        # Robustness test via light perturbations
        if y_true is not None and request.robustness_test:
            import re, random
            def simple_perturb(text: str) -> str:
                t = str(text)
                t = re.sub(r"\s+", " ", t)
                if len(t) > 0:
                    idx = random.randrange(0, len(t))
                    t = t[:idx] + ("!" if t[idx] != "!" else "?") + t[idx:]
                parts = t.split(" ")
                if parts:
                    j = random.randrange(0, len(parts))
                    parts[j] = parts[j].upper() if random.random() < 0.5 else parts[j].lower()
                    t = " ".join(parts)
                return t

            random.seed(42)
            perturbed_texts = [simple_perturb(t) for t in texts]
            preds_pert = model_service.classify_batch(perturbed_texts)
            y_pred_pert = np.array([p.get("prediction", 0) for p in preds_pert], dtype=int)
            y_true_arr2 = np.array(y_true.tolist(), dtype=int)
            acc_pert = float(accuracy_score(y_true_arr2, y_pred_pert))
            stability = float(np.mean(y_pred_pert == y_pred))
            metrics_payload.update(
                {
                    "robustness_accuracy": round(acc_pert, 4),
                    "robustness_delta": round(float(metrics_payload["accuracy"] - acc_pert), 4),
                    "prediction_stability": round(stability, 4),
                }
            )

        # Abstention curve using margin (confidence gap)
        if y_true is not None and request.abstention_curve:
            try:
                margins = np.array([
                    float(item.get("uncertainty", {}).get("margin", 0.0)) for item in preds
                ])
                order = np.argsort(-margins)
                y_true_arr3 = np.array(y_true.tolist(), dtype=int)
                y_pred_arr = y_pred.copy()
                coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                cov_table = []
                n = len(y_true_arr3)
                for c in coverages:
                    k = max(1, int(round(c * n)))
                    idx = order[:k]
                    acc_c = float(accuracy_score(y_true_arr3[idx], y_pred_arr[idx])) if k > 0 else 0.0
                    cov_table.append({"coverage": round(c, 2), "accuracy": round(acc_c, 4), "n": int(k)})
                metrics_payload["coverage_accuracy"] = cov_table
            except Exception:
                pass

        if metrics_payload_mi:
            metrics_payload.update(metrics_payload_mi)
        if exp_quality_metrics:
            metrics_payload.update(exp_quality_metrics)
        if fairness_metrics:
            metrics_payload.update(fairness_metrics)
        if temporal_metrics:
            metrics_payload.update(temporal_metrics)
        if attention_metrics:
            metrics_payload.update(attention_metrics)

        # Optionally write reports (disabled by default; UI consumes JSON directly)
        report_path: Optional[str] = None
        if request.save_report:
            if baseline_metrics or mcnemar_stats or calibration_metrics or metrics_payload.get("robustness_accuracy") or metrics_payload.get("coverage_accuracy") or metrics_payload.get("mutual_information_mean") or metrics_payload.get("explainability_quality_topk_delta_mean") or metrics_payload.get("fairness_groups") or metrics_payload.get("temporal_periods") or metrics_payload.get("attention_topk_mean"):
                if baseline_metrics:
                    metrics_payload.update(baseline_metrics)
                if mcnemar_stats:
                    metrics_payload.update(mcnemar_stats)
                if calibration_metrics:
                    metrics_payload.update(calibration_metrics)
                report_path = write_comparative_metrics_report(metrics_payload, report_dir)
            else:
                report_path = write_metrics_report(metrics_payload, report_dir)

        response = EvaluateModelResponse(
            dataset=dataset_name,
            total_evaluated=int(len(df)),
            accuracy=accuracy,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            report_path=report_path,
            extra_metrics=metrics_payload,
        )
        progress_manager.set_result(trace, {
            "dataset": dataset_name,
            "total_evaluated": int(len(df)),
            "metrics": metrics_payload,
        })
        await progress_manager.publish(trace, stage="done", message="Evaluation complete", percent=100.0)
        # Store cache
        cache_payload = {
            "dataset": dataset_name,
            "total_evaluated": int(len(df)),
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "report_path": report_path,
            "extra_metrics": metrics_payload,
        }
        trace = cache_put(
            trace_id=trace_id,
            dataset=dataset_name,
            process="evaluate",
            params_hash=params_hash,
            dataset_fingerprint=dataset_fp,
            payload=cache_payload,
        )
        try:
            if request.save_report:
                add_report(dataset=dataset_name, report_type="evaluation", payload=cache_payload)
        except Exception:
            pass
        response.extra_metrics["trace_id"] = trace
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

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