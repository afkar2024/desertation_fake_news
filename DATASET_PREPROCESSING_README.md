# Dataset Management and Preprocessing System

This system implements the data sources and preprocessing pipeline as described in Point 8 of the pilot report.

## Overview

The system provides:
- **Dataset Management**: Download and manage multiple datasets (LIAR, PolitiFact, FakeNewsNet)
- **Preprocessing Pipeline**: Complete text preprocessing with feature extraction
- **API Endpoints**: RESTful API for dataset operations
- **CLI Interface**: Command-line tools for batch operations

## Datasets Supported

### 1. LIAR Dataset
- **Source**: 12,836 political statements with 6 veracity categories
- **Labels**: true, mostly-true, half-true, mostly-false, false, pants-fire
- **Format**: TSV files (train, test, validation)

### 2. PolitiFact Dataset
- **Source**: ~8,000 fact-checked articles (simulated for now)
- **Labels**: Same as LIAR dataset
- **Format**: JSON

### 3. FakeNewsNet Dataset
- **Source**: Multimodal corpus with text, user profiles, network features
- **Format**: JSON (requires manual download from GitHub)

## Preprocessing Pipeline

Following the pilot report specifications:

### 1. Normalization
- Lowercase conversion
- HTML tag removal
- Unicode normalization
- URL/email/phone number replacement

### 2. Stopword Removal & Cleaning
- Remove common stopwords
- Remove special characters
- Clean whitespace

### 3. Tokenization
- HuggingFace WordPiece tokenizer (preserves subword units)
- Alternative NLTK tokenizer

### 4. Feature Engineering
- **Readability**: Flesch-Kincaid grade, reading ease
- **Sentiment**: Polarity, subjectivity scores
- **Linguistic**: Exclamation marks, caps ratio, etc.
- **Metadata**: Source credibility, publication timing

### 5. Balancing & Augmentation
- Undersample majority classes
- Oversample minority classes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (automatically handled):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### API Endpoints

Start the server:
```bash
python start_server.py
```

Available endpoints:
- `GET /datasets/` - List all datasets
- `POST /datasets/download` - Download all datasets
- `POST /datasets/download/{dataset_name}` - Download specific dataset
- `GET /datasets/{dataset_name}/info` - Get dataset info
- `GET /datasets/{dataset_name}/sample` - Get sample records
- `POST /datasets/{dataset_name}/preprocess` - Preprocess dataset
- `POST /datasets/preprocess/text` - Preprocess single text

### CLI Interface

```bash
# Download all datasets
python download_and_preprocess.py --download

# Download specific dataset
python download_and_preprocess.py --dataset liar
python download_and_preprocess.py --dataset politifact

# Preprocess dataset
python download_and_preprocess.py --preprocess liar --balance undersample

# Preprocess with custom output
python download_and_preprocess.py --preprocess liar --output ./my_data/
```

### Python API

```python
from app.dataset_manager import dataset_manager
from app.preprocessing import preprocessor

# Download datasets
results = dataset_manager.download_all_datasets()

# Load dataset
df = dataset_manager.load_liar_dataset()

# Preprocess
processed_df = preprocessor.preprocess_dataframe(df, 'statement', 'label')

# Balance dataset
balanced_df = preprocessor.balance_dataset(processed_df, 'label_encoded', 'undersample')

# Save processed data
preprocessor.save_preprocessed_data(balanced_df, 'processed_liar.csv')
```

## Example Workflow

1. **Download datasets**:
```bash
curl -X POST http://localhost:8000/datasets/download
```

2. **Check dataset status**:
```bash
curl http://localhost:8000/datasets/
```

3. **Get sample data**:
```bash
curl http://localhost:8000/datasets/liar/sample?limit=5
```

4. **Preprocess dataset**:
```bash
curl -X POST http://localhost:8000/datasets/liar/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "liar",
    "text_column": "statement",
    "label_column": "label",
    "balance_strategy": "undersample"
  }'
```

5. **Preprocess single text**:
```bash
curl -X POST http://localhost:8000/datasets/preprocess/text \
  -H "Content-Type: application/json" \
  -d '"This is a sample news statement to analyze."'
```

## File Structure

```
├── app/
│   ├── dataset_manager.py     # Dataset downloading and management
│   ├── preprocessing.py       # Text preprocessing pipeline
│   ├── dataset_api.py        # API endpoints
│   └── main.py               # Main FastAPI app
├── datasets/                 # Downloaded datasets
│   ├── liar/
│   ├── politifact/
│   └── fakenewsnet/
├── processed_data/           # Preprocessed datasets
└── download_and_preprocess.py # CLI script
```

## Features Generated

The preprocessing pipeline generates these features:

### Text Features
- `processed_text`: Cleaned and tokenized text
- `flesch_kincaid_grade`: Readability grade level
- `flesch_reading_ease`: Reading ease score
- `sentiment_polarity`: Sentiment polarity (-1 to 1)
- `sentiment_subjectivity`: Subjectivity score (0 to 1)

### Linguistic Features
- `exclamation_count`: Number of exclamation marks
- `question_count`: Number of question marks
- `caps_ratio`: Ratio of uppercase characters
- `all_caps_words`: Number of all-caps words
- `word_count`: Total word count
- `sentence_count`: Total sentence count

### Metadata Features
- `source_credibility`: Binary credibility score
- `publication_hour`: Hour of publication
- `publication_day_of_week`: Day of week
- `has_author`: Whether author is present

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start server**: `python start_server.py`
3. **Download datasets**: Use API or CLI
4. **Run preprocessing**: Process datasets with full pipeline
5. **Train models**: Use processed data for ML model training

## Notes

- The system handles large datasets efficiently with background processing
- All preprocessing steps are configurable
- Generated features are saved with the processed data
- Label mappings are preserved for model training
- The system is designed to be extensible for additional datasets 