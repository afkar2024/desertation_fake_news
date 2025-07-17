# Fake News Detection Dataset Report

**Generated:** 2025-07-17 09:45:08
**Report Type:** Full_Pipeline_Background
**Dataset:** POLITIFACT
**Output Directory:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data`

---

## 📥 Download Results

- **Total Datasets:** 1
- **Successful:** 1 ✅
- **Failed:** 0 ❌

### Dataset Status

| Dataset | Status | Location | Description |
|---------|--------|----------|-------------|
| politifact | ✅ Success | `N/A` | N/A |

---

## 🔄 Preprocessing Results

### Dataset Overview

| Metric | Original | Processed | Balanced |
|--------|----------|-----------|----------|
| **Rows** | 100 | 100 | 96 |
| **Columns** | 8 | 33 | 33 |
| **Features Added** | - | 25 | - |

### 📁 Generated Files

- **Processed Dataset:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/politifact_processed.csv`
- **Label Mapping:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/politifact_processed_label_mapping.json`
- **This Report:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/full_pipeline_background_report_politifact_20250717_094508.md`

### 🆕 New Features Added

| Feature | Type | Description |
|---------|------|-------------|
| `processed_text` | Other | Generated feature |
| `flesch_kincaid_grade` | Readability | Flesch-Kincaid grade level score |
| `flesch_reading_ease` | Readability | Flesch reading ease score (0-100) |
| `sentence_count` | Other | Generated feature |
| `word_count` | Text Stats | Number of words |
| `avg_sentence_length` | Other | Generated feature |
| `avg_word_length` | Text Stats | Average word length |
| `sentiment_polarity` | Sentiment | Sentiment polarity (-1 to 1) |
| `sentiment_subjectivity` | Sentiment | Sentiment subjectivity (0 to 1) |
| `sentiment_label` | Sentiment | Categorical sentiment (positive/negative/neutral) |
| `exclamation_count` | Linguistic | Number of exclamation marks |
| `question_count` | Linguistic | Number of question marks |
| `caps_count` | Other | Generated feature |
| `digit_count` | Other | Generated feature |
| `caps_ratio` | Linguistic | Ratio of uppercase characters |
| `digit_ratio` | Other | Generated feature |
| `all_caps_words` | Linguistic | Number of all-caps words |
| `text_length` | Other | Generated feature |
| `source_credibility` | Other | Generated feature |
| `publication_hour` | Other | Generated feature |
| `publication_day_of_week` | Other | Generated feature |
| `publication_month` | Other | Generated feature |
| `has_author` | Other | Generated feature |
| `author_length` | Other | Generated feature |
| `label_encoded` | Processing | Numerically encoded labels |

### 🏷️ Label Analysis

| Label | Count | Percentage |
|-------|-------|------------|
| true | 17 | 17.0% |
| mostly-true | 17 | 17.0% |
| half-true | 17 | 17.0% |
| mostly-false | 17 | 17.0% |
| false | 16 | 16.0% |
| pants-fire | 16 | 16.0% |

**Total Labels:** 6
**Most Common:** true (17 samples)
**Least Common:** pants-fire (16 samples)

#### Label Mapping

| Original Label | Encoded Value |
|---------------|---------------|
| false | 0 |
| half-true | 1 |
| mostly-false | 2 |
| mostly-true | 3 |
| pants-fire | 4 |
| true | 5 |

### 📊 Feature Analysis

#### Readability Metrics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Flesch Kincaid Grade | 13.10 | 0.00 | 13.10 | 13.10 |
| Flesch Reading Ease | 8.20 | 0.00 | 8.20 | 8.20 |

#### Sentiment Analysis

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Polarity | 0.000 | 0.000 | 0.000 | 0.000 |
| Subjectivity | 0.100 | 0.000 | 0.100 | 0.100 |

##### Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| neutral | 100 | 100.0% |

#### Linguistic Features

| Feature | Mean | Std Dev | Max |
|---------|------|---------|-----|
| Exclamation Count | 0.000 | 0.000 | 0 |
| Question Count | 0.000 | 0.000 | 0 |
| Caps Ratio | 0.035 | 0.000 | 0 |
| All Caps Words | 0.000 | 0.000 | 0 |

#### Text Statistics

| Column | Avg Length | Max Length | Min Length | Std Dev |
|--------|------------|------------|------------|----------|
| statement | 29 | 29 | 28 | 0 |

### ⚖️ Dataset Balancing

| Label | Original Count | Balanced Count | Change |
|-------|---------------|----------------|--------|
| true | 17 | 16 | -1 |
| mostly-true | 17 | 16 | -1 |
| half-true | 17 | 16 | -1 |
| mostly-false | 17 | 16 | -1 |
| false | 16 | 16 | 0 |
| pants-fire | 16 | 16 | 0 |

**Balancing Strategy Applied:** Successfully reduced dataset size while maintaining label distribution
**Original Size:** 100 samples
**Balanced Size:** 96 samples
**Size Reduction:** 4 samples (4.0%)

---

## 📋 How to Use These Files

### Loading the Processed Dataset

```python
import pandas as pd
import json

# Load the processed dataset
df = pd.read_csv('/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/politifact_processed.csv')

# Load the label mapping
with open('/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/politifact_processed_label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

print(f'Dataset shape: {df.shape}')
print(f'Label mapping: {label_mapping}')
```

### Key Columns for Machine Learning

- **Text Features:** `statement`
- **Numerical Features:** `flesch_kincaid_grade`, `word_count`, `sentiment_polarity`
- **Target Variable:** `label_encoded` (numerical) or `label` (categorical)

### Next Steps

1. **Data Exploration:** Load the dataset and explore the new features
2. **Feature Selection:** Choose the most relevant features for your model
3. **Model Training:** Use the processed features to train your fake news detection model
4. **Evaluation:** Test your model using the balanced dataset

---

*Report generated by Fake News Detection Dataset Processing Pipeline*
*Timestamp: 2025-07-17T09:45:08.051102*
