# Fake News Detection Dataset Report

**Generated:** 2025-07-17 09:35:33
**Report Type:** Preprocessing
**Dataset:** LIAR
**Output Directory:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data`

---

## 🔄 Preprocessing Results

### Dataset Overview

| Metric | Original | Processed | Balanced |
|--------|----------|-----------|----------|
| **Rows** | 12,791 | 12,791 | 6,282 |
| **Columns** | 15 | 40 | 40 |
| **Features Added** | - | 25 | - |

### 📁 Generated Files

- **Processed Dataset:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/liar_processed.csv`
- **Label Mapping:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/liar_processed_label_mapping.json`
- **This Report:** `/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/preprocessing_report_liar_20250717_093533.md`

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
| half-true | 2,627 | 20.5% |
| false | 2,507 | 19.6% |
| mostly-true | 2,454 | 19.2% |
| barely-true | 2,103 | 16.4% |
| true | 2,053 | 16.1% |
| pants-fire | 1,047 | 8.2% |

**Total Labels:** 6
**Most Common:** half-true (2,627 samples)
**Least Common:** pants-fire (1,047 samples)

#### Label Mapping

| Original Label | Encoded Value |
|---------------|---------------|
| barely-true | 0 |
| false | 1 |
| half-true | 2 |
| mostly-true | 3 |
| pants-fire | 4 |
| true | 5 |

### 📊 Feature Analysis

#### Readability Metrics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Flesch Kincaid Grade | 8.35 | 3.64 | -2.70 | 26.50 |
| Flesch Reading Ease | 62.95 | 20.11 | -91.30 | 119.19 |

#### Sentiment Analysis

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Polarity | 0.050 | 0.211 | -1.000 | 1.000 |
| Subjectivity | 0.266 | 0.274 | 0.000 | 1.000 |

##### Sentiment Distribution

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| neutral | 7,785 | 60.9% |
| positive | 3,480 | 27.2% |
| negative | 1,526 | 11.9% |

#### Linguistic Features

| Feature | Mean | Std Dev | Max |
|---------|------|---------|-----|
| Exclamation Count | 0.007 | 0.103 | 5 |
| Question Count | 0.008 | 0.096 | 4 |
| Caps Ratio | 0.037 | 0.027 | 1 |
| All Caps Words | 0.082 | 0.356 | 11 |

#### Text Statistics

| Column | Avg Length | Max Length | Min Length | Std Dev |
|--------|------------|------------|------------|----------|
| statement | 107 | 3192 | 11 | 63 |

### ⚖️ Dataset Balancing

| Label | Original Count | Balanced Count | Change |
|-------|---------------|----------------|--------|
| half-true | 2,627 | 1,047 | -1580 |
| false | 2,507 | 1,047 | -1460 |
| mostly-true | 2,454 | 1,047 | -1407 |
| barely-true | 2,103 | 1,047 | -1056 |
| true | 2,053 | 1,047 | -1006 |
| pants-fire | 1,047 | 1,047 | 0 |

**Balancing Strategy Applied:** Successfully reduced dataset size while maintaining label distribution
**Original Size:** 12,791 samples
**Balanced Size:** 6,282 samples
**Size Reduction:** 6,509 samples (50.9%)

---

## 📋 How to Use These Files

### Loading the Processed Dataset

```python
import pandas as pd
import json

# Load the processed dataset
df = pd.read_csv('/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/liar_processed.csv')

# Load the label mapping
with open('/home/kishan/kishan/projects/afkar_desertation/desertation_fake_news/processed_data/liar_processed_label_mapping.json', 'r') as f:
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
*Timestamp: 2025-07-17T09:35:33.422635*
