# Fake News Detection API - Full Pipeline Usage Guide

## Overview

The Fake News Detection API now includes comprehensive endpoints that handle the entire dataset processing pipeline: download, preprocessing, balancing, and markdown report generation. This guide shows you how to use these endpoints effectively.

## Available Endpoints

### 1. Full Pipeline (Synchronous)
**Endpoint:** `POST /datasets/full-pipeline/{dataset_name}`

This endpoint processes the entire pipeline synchronously and returns a markdown report file for download.

### 2. Full Pipeline (Background)
**Endpoint:** `POST /datasets/full-pipeline/{dataset_name}/background`

This endpoint starts the pipeline in the background and returns immediately. The markdown report is saved to the `processed_data` directory.

## Supported Datasets

- **liar**: LIAR dataset (12,791 political statements)
- **politifact**: PolitiFact dataset (~8,000 fact-checked articles)
- **fakenewsnet**: FakeNewsNet dataset (multimodal corpus)

## API Request Format

### Request Body Parameters

```json
{
  "text_column": "statement",           // Column containing text to analyze
  "label_column": "label",              // Column containing labels
  "balance_strategy": "undersample",    // Optional: "undersample", "oversample", or null
  "download_if_missing": true,          // Whether to download dataset if not present
  "return_markdown": true               // Whether to return markdown report
}
```

### Parameters Explained

- **text_column**: The column name containing the text to be processed (default: "statement")
- **label_column**: The column name containing the labels (default: "label")
- **balance_strategy**: 
  - `"undersample"`: Reduces majority classes to match minority class size
  - `"oversample"`: Increases minority classes to match majority class size
  - `null`: No balancing applied
- **download_if_missing**: If `true`, downloads the dataset if it's not already present
- **return_markdown**: If `true`, returns a markdown report file; if `false`, returns JSON summary

## Usage Examples

### Example 1: Process LIAR Dataset with Undersampling

```bash
curl -X POST "http://localhost:8000/datasets/full-pipeline/liar" \
  -H "Content-Type: application/json" \
  -d '{
    "balance_strategy": "undersample",
    "download_if_missing": false,
    "return_markdown": true
  }' \
  --output liar_report.md
```

### Example 2: Process PolitiFact Dataset in Background

```bash
curl -X POST "http://localhost:8000/datasets/full-pipeline/politifact/background" \
  -H "Content-Type: application/json" \
  -d '{
    "balance_strategy": "undersample",
    "download_if_missing": true,
    "return_markdown": true
  }'
```

### Example 3: Process Without Balancing (JSON Response)

```bash
curl -X POST "http://localhost:8000/datasets/full-pipeline/liar" \
  -H "Content-Type: application/json" \
  -d '{
    "return_markdown": false
  }'
```

## Python Examples

### Using requests library

```python
import requests

# Synchronous processing with markdown report
response = requests.post(
    "http://localhost:8000/datasets/full-pipeline/liar",
    json={
        "balance_strategy": "undersample",
        "download_if_missing": True,
        "return_markdown": True
    }
)

if response.status_code == 200:
    # Save the markdown report
    with open("liar_report.md", "wb") as f:
        f.write(response.content)
    print("Markdown report saved as liar_report.md")
else:
    print(f"Error: {response.json()}")

# Background processing
response = requests.post(
    "http://localhost:8000/datasets/full-pipeline/politifact/background",
    json={
        "balance_strategy": "undersample",
        "download_if_missing": True,
        "return_markdown": True
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Background task started: {result['message']}")
    print(f"Estimated time: {result['estimated_time']}")
```

## Response Formats

### Synchronous Endpoint (return_markdown: true)
Returns a downloadable markdown file with comprehensive analysis.

### Synchronous Endpoint (return_markdown: false)
Returns JSON response:

```json
{
  "success": true,
  "message": "Full pipeline completed for liar",
  "download_results": {"liar": true},
  "original_records": 12791,
  "processed_records": 12791,
  "final_records": 6282,
  "features_added": 25,
  "balance_strategy": "undersample",
  "files_generated": {
    "processed_csv": "/path/to/liar_processed.csv",
    "label_mapping": "/path/to/liar_processed_label_mapping.json"
  }
}
```

### Background Endpoint
Returns JSON response:

```json
{
  "success": true,
  "message": "Full pipeline started for politifact in background",
  "dataset": "politifact",
  "estimated_time": "2-5 minutes depending on dataset size"
}
```

## Generated Files

The pipeline generates several files in the `processed_data` directory:

1. **Processed Dataset**: `{dataset_name}_processed.csv`
   - Contains all original data plus new features
   - Ready for machine learning

2. **Label Mapping**: `{dataset_name}_processed_label_mapping.json`
   - Maps original labels to numerical encodings
   - Useful for model interpretation

3. **Markdown Report**: `full_pipeline_report_{dataset_name}_{timestamp}.md`
   - Comprehensive analysis and statistics
   - Usage instructions and next steps

## Markdown Report Contents

The generated markdown report includes:

### 📥 Download Results
- Dataset download status and locations
- Success/failure information

### 🔄 Preprocessing Results
- Dataset overview (original vs processed vs balanced)
- File locations for all generated files
- New features added with descriptions

### 🏷️ Label Analysis
- Label distribution and percentages
- Label mapping (original to encoded values)
- Most/least common labels

### 📊 Feature Analysis
- **Readability Metrics**: Flesch-Kincaid, reading ease, etc.
- **Sentiment Analysis**: Polarity, subjectivity, distribution
- **Linguistic Features**: Exclamation marks, caps ratio, etc.
- **Text Statistics**: Length statistics for text columns

### ⚖️ Dataset Balancing
- Before/after label distributions
- Size reduction information
- Balancing strategy details

### 📋 Usage Instructions
- Code examples for loading processed data
- Key columns for machine learning
- Next steps for model development

## Error Handling

Common error responses:

```json
{
  "detail": "Dataset not found"
}
```

```json
{
  "detail": "Could not load liar dataset"
}
```

```json
{
  "detail": "Text column 'statement' not found"
}
```

## Performance Considerations

- **Synchronous processing**: Best for smaller datasets or when you need immediate results
- **Background processing**: Best for larger datasets or when you don't need immediate results
- **Estimated processing times**:
  - LIAR dataset: 2-3 minutes
  - PolitiFact dataset: 1-2 minutes
  - FakeNewsNet dataset: 5-10 minutes

## Next Steps

After processing your dataset:

1. **Load the processed data** using the provided code examples
2. **Explore the features** using the markdown report analysis
3. **Train your model** using the numerical features and encoded labels
4. **Evaluate performance** using the balanced dataset

## Server Management

To start the server:
```bash
python start_server.py
```

To stop the server:
```bash
python stop_server.py
```

The server runs on `http://localhost:8000` by default.

## API Documentation

For complete API documentation, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 