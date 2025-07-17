"""
Text Preprocessing Pipeline for Fake News Detection
Implements all preprocessing steps mentioned in the pilot report.
"""

import re
import html
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import warnings
from collections import Counter

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="textstat")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import textstat  # type: ignore
from textblob import TextBlob  # type: ignore
from transformers import AutoTokenizer
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline for fake news detection"""
    
    def __init__(self, language: str = "english", model_name: str = "bert-base-uncased"):
        self.language = language
        self.model_name = model_name
        
        # Initialize components
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        
        # Custom stop words for news domain
        self.custom_stop_words = {
            'said', 'says', 'according', 'reported', 'reports', 'news', 'article',
            'story', 'source', 'sources', 'journalist', 'reporter', 'editor',
            'breaking', 'update', 'latest', 'exclusive', 'developing'
        }
        
        self.stop_words.update(self.custom_stop_words)
        
        # Regex patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?;:\'"()-]')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        if not text:
            return ""
        
        # Normalize Unicode to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, emails, phones, and special characters"""
        if not text:
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub(' [URL] ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' [EMAIL] ', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub(' [PHONE] ', text)
        
        # Remove special characters (keep basic punctuation)
        text = self.special_chars_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
    
    def normalize_text(self, text: str) -> str:
        """Apply normalization steps as mentioned in pilot report"""
        if not text:
            return ""
        
        # Step 1: Lowercase conversion
        text = text.lower()
        
        # Step 2: HTML tag removal
        text = self.remove_html_tags(text)
        
        # Step 3: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 4: Clean text
        text = self.clean_text(text)
        
        return text
    
    def tokenize_text(self, text: str, use_huggingface: bool = True) -> List[str]:
        """Tokenize text using HuggingFace WordPiece tokenizer or NLTK"""
        if not text:
            return []
        
        if use_huggingface:
            # Use HuggingFace tokenizer (preserves subword units)
            tokens = self.tokenizer.tokenize(text)
            return tokens
        else:
            # Use NLTK tokenizer
            tokens = word_tokenize(text)
            return tokens
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, 
                       lemmatize: bool = True, use_huggingface_tokenizer: bool = True) -> str:
        """Complete preprocessing pipeline for a single text"""
        
        # Step 1: Normalization
        text = self.normalize_text(text)
        
        # Step 2: Tokenization
        tokens = self.tokenize_text(text, use_huggingface=use_huggingface_tokenizer)
        
        if not use_huggingface_tokenizer:
            # Step 3: Stopword removal (only for NLTK tokens)
            if remove_stopwords:
                tokens = self.remove_stopwords(tokens)
            
            # Step 4: Lemmatization (only for NLTK tokens)
            if lemmatize:
                tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back to text
        return ' '.join(tokens)
    
    def calculate_readability_features(self, text: str) -> Dict[str, float]:
        """Calculate readability indices (Flesch-Kincaid)"""
        if not text or len(text.strip()) < 10:
            return {
                'flesch_kincaid_grade': 0.0,
                'flesch_reading_ease': 0.0,
                'sentence_count': 0,
                'word_count': 0,
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0
            }
        
        try:
            # Calculate readability scores using textstat
            fk_grade = textstat.flesch_kincaid_grade(text)  # type: ignore
            fk_ease = textstat.flesch_reading_ease(text)  # type: ignore
            
            # Calculate basic statistics
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            sentence_count = len(sentences)
            word_count = len(words)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            
            return {
                'flesch_kincaid_grade': float(fk_grade),
                'flesch_reading_ease': float(fk_ease),
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length
            }
        except Exception as e:
            logger.warning(f"Error calculating readability features: {e}")
            return {
                'flesch_kincaid_grade': 0.0,
                'flesch_reading_ease': 0.0,
                'sentence_count': 0,
                'word_count': 0,
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0
            }
    
    def calculate_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Calculate sentiment polarity scores"""
        if not text:
            return {
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0,
                'sentiment_label': 'neutral'
            }
        
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment  # type: ignore
            polarity = float(sentiment.polarity)  # type: ignore
            subjectivity = float(sentiment.subjectivity)  # type: ignore
            
            # Determine sentiment label
            if polarity > 0.1:
                sentiment_label = 'positive'
            elif polarity < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'sentiment_polarity': polarity,
                'sentiment_subjectivity': subjectivity,
                'sentiment_label': sentiment_label
            }
        except Exception as e:
            logger.warning(f"Error calculating sentiment features: {e}")
            return {
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0,
                'sentiment_label': 'neutral'
            }
    
    def calculate_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Calculate additional linguistic features"""
        if not text:
            return {}
        
        try:
            # Count various linguistic features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_count = sum(1 for c in text if c.isupper())
            digit_count = sum(1 for c in text if c.isdigit())
            
            # Calculate ratios
            text_length = len(text)
            caps_ratio = caps_count / text_length if text_length > 0 else 0
            digit_ratio = digit_count / text_length if text_length > 0 else 0
            
            # Count specific patterns
            all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
            
            return {
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'caps_count': caps_count,
                'digit_count': digit_count,
                'caps_ratio': caps_ratio,
                'digit_ratio': digit_ratio,
                'all_caps_words': all_caps_words,
                'text_length': text_length
            }
        except Exception as e:
            logger.warning(f"Error calculating linguistic features: {e}")
            return {}
    
    def extract_metadata_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata features from a data row"""
        features = {}
        
        # Source credibility (placeholder - would need actual credibility database)
        source = row.get('source', '') or ''
        trusted_sources = {
            'reuters', 'ap', 'bbc', 'cnn', 'nytimes', 'washingtonpost',
            'guardian', 'npr', 'pbs', 'wsj'
        }
        
        features['source_credibility'] = 1 if any(src in source.lower() for src in trusted_sources) else 0
        
        # Publication date features
        pub_date_col = row.get('published_date')
        if pub_date_col is not None and pd.notna(pub_date_col):
            try:
                pub_date = pd.to_datetime(pub_date_col)
                # Handle both single values and series
                if hasattr(pub_date, 'hour'):
                    features['publication_hour'] = int(pub_date.hour)  # type: ignore
                    features['publication_day_of_week'] = int(pub_date.dayofweek)  # type: ignore
                    features['publication_month'] = int(pub_date.month)  # type: ignore
                elif hasattr(pub_date, 'dt'):
                    # Handle series case
                    features['publication_hour'] = int(pub_date.dt.hour.iloc[0])  # type: ignore
                    features['publication_day_of_week'] = int(pub_date.dt.dayofweek.iloc[0])  # type: ignore
                    features['publication_month'] = int(pub_date.dt.month.iloc[0])  # type: ignore
                else:
                    # Fallback
                    features['publication_hour'] = 0
                    features['publication_day_of_week'] = 0
                    features['publication_month'] = 0
            except Exception:
                features['publication_hour'] = 0
                features['publication_day_of_week'] = 0
                features['publication_month'] = 0
        else:
            features['publication_hour'] = 0
            features['publication_day_of_week'] = 0
            features['publication_month'] = 0
        
        # Author features
        author = row.get('author', '')
        features['has_author'] = 1 if author and author.strip() else 0
        features['author_length'] = len(author) if author else 0
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'statement', 
                           label_column: str = 'label') -> pd.DataFrame:
        """Preprocess an entire DataFrame"""
        logger.info(f"Starting preprocessing of {len(df)} records...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Preprocess text
        logger.info("Preprocessing text...")
        processed_df['processed_text'] = processed_df[text_column].apply(
            lambda x: self.preprocess_text(str(x) if pd.notna(x) else "")
        )
        
        # Calculate readability features
        logger.info("Calculating readability features...")
        readability_features = processed_df[text_column].apply(
            lambda x: self.calculate_readability_features(str(x) if pd.notna(x) else "")
        )
        readability_df = pd.DataFrame(readability_features.tolist())
        processed_df = pd.concat([processed_df, readability_df], axis=1)
        
        # Calculate sentiment features
        logger.info("Calculating sentiment features...")
        sentiment_features = processed_df[text_column].apply(
            lambda x: self.calculate_sentiment_features(str(x) if pd.notna(x) else "")
        )
        sentiment_df = pd.DataFrame(sentiment_features.tolist())
        processed_df = pd.concat([processed_df, sentiment_df], axis=1)
        
        # Calculate linguistic features
        logger.info("Calculating linguistic features...")
        linguistic_features = processed_df[text_column].apply(
            lambda x: self.calculate_linguistic_features(str(x) if pd.notna(x) else "")
        )
        linguistic_df = pd.DataFrame(linguistic_features.tolist())
        processed_df = pd.concat([processed_df, linguistic_df], axis=1)
        
        # Extract metadata features
        logger.info("Extracting metadata features...")
        metadata_features = processed_df.apply(self.extract_metadata_features, axis=1)
        metadata_df = pd.DataFrame(metadata_features.tolist())
        processed_df = pd.concat([processed_df, metadata_df], axis=1)
        
        # Encode labels
        if label_column in processed_df.columns:
            logger.info("Encoding labels...")
            processed_df['label_encoded'] = self.label_encoder.fit_transform(
                processed_df[label_column].fillna('unknown')
            )
            
            # Save label mapping with proper type handling
            if self.label_encoder.classes_ is not None:
                label_mapping = {
                    str(cls): int(self.label_encoder.transform([cls])[0])
                    for cls in self.label_encoder.classes_
                }
                processed_df.attrs['label_mapping'] = label_mapping
        
        logger.info(f"Preprocessing completed. Final shape: {processed_df.shape}")
        return processed_df
    
    def balance_dataset(self, df: pd.DataFrame, label_column: str = 'label_encoded', 
                       strategy: str = 'undersample') -> pd.DataFrame:
        """Balance dataset using undersampling or oversampling"""
        logger.info(f"Balancing dataset using {strategy} strategy...")
        
        if label_column not in df.columns:
            logger.warning(f"Label column {label_column} not found. Skipping balancing.")
            return df
        
        # Get class distribution
        class_counts = df[label_column].value_counts()
        logger.info(f"Original class distribution:\n{class_counts}")
        
        if strategy == 'undersample':
            # Undersample majority classes
            min_class_size = class_counts.min()
            
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df[df[label_column] == class_label]
                if len(class_df) > min_class_size:
                    class_df = resample(class_df, n_samples=min_class_size, random_state=42)
                balanced_dfs.append(class_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif strategy == 'oversample':
            # Oversample minority classes
            max_class_size = class_counts.max()
            
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df[df[label_column] == class_label]
                if len(class_df) < max_class_size:
                    class_df = resample(class_df, n_samples=max_class_size, 
                                      random_state=42, replace=True)
                balanced_dfs.append(class_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        else:
            logger.warning(f"Unknown balancing strategy: {strategy}")
            return df
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log new distribution
        new_class_counts = balanced_df[label_column].value_counts()
        logger.info(f"Balanced class distribution:\n{new_class_counts}")
        
        return balanced_df
    
    def save_preprocessed_data(self, df: pd.DataFrame, filepath: str):
        """Save preprocessed data to file"""
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath_obj.suffix == '.csv':
            df.to_csv(filepath_obj, index=False)
        elif filepath_obj.suffix == '.json':
            df.to_json(filepath_obj, orient='records', indent=2)
        elif filepath_obj.suffix == '.pkl':
            df.to_pickle(filepath_obj)
        else:
            logger.warning(f"Unsupported file format: {filepath_obj.suffix}")
            return
        
        logger.info(f"Saved preprocessed data to {filepath_obj}")
        
        # Save label mapping if available
        if hasattr(df, 'attrs') and 'label_mapping' in df.attrs:
            mapping_file = filepath_obj.parent / f"{filepath_obj.stem}_label_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(df.attrs['label_mapping'], f, indent=2)
            logger.info(f"Saved label mapping to {mapping_file}")


# Global preprocessor instance
preprocessor = TextPreprocessor() 