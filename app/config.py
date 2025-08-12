"""
Configuration settings for the Fake News Detector API
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    reload: bool = True

    # API Configuration
    app_name: str = "Adaptive Fake News Detector"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Configuration (for future use)
    database_url: Optional[str] = None
    
    # External API Keys
    newsapi_key: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    
    # Data Collection Settings
    max_articles_per_source: int = 100
    collection_interval_hours: int = 6
    
    # Prediction Settings
    confidence_threshold: float = 0.7
    enable_explanations: bool = True
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    explanation_max_evals: int = 200
    explanation_enabled_classes: int = 2
    model_path: Optional[str] = None  # If set, load local fine-tuned model
    
    # Rate Limiting
    requests_per_minute: int = 60
    
    # Logging
    log_level: str = "info"
    log_file: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Silence pydantic protected namespace warning for fields starting with model_
        protected_namespaces = ("settings_",)

# Global settings instance
settings = Settings()

# Predefined data sources configuration
DEFAULT_SOURCES = [
    {
        "name": "NewsAPI General",
        "type": "api",
        "url": "https://newsapi.org/v2/top-headlines",
        "params": {"country": "us", "category": "general", "pageSize": 20},
        "api_key_required": True,
        "rate_limit": 1000,  # requests per day
        "is_active": False
    },
    {
        "name": "NewsAPI Technology",
        "type": "api", 
        "url": "https://newsapi.org/v2/top-headlines",
        "params": {"country": "us", "category": "technology", "pageSize": 20},
        "api_key_required": True,
        "rate_limit": 1000,
        "is_active": False
    },
    {
        "name": "BBC RSS World",
        "type": "rss",
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "api_key_required": False,
        "rate_limit": None,
        "is_active": True
    },
    {
        "name": "BBC RSS Technology",
        "type": "rss",
        "url": "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "api_key_required": False,
        "rate_limit": None,
        "is_active": True
    },
    {
        "name": "CNN RSS",
        "type": "rss",
        "url": "http://rss.cnn.com/rss/edition.rss",
        "api_key_required": False,
        "rate_limit": None,
        "is_active": True
    },
    {
        "name": "Reuters World News",
        "type": "rss",
        "url": "https://feeds.reuters.com/reuters/worldNews",
        "api_key_required": False,
        "rate_limit": None,
        "is_active": True
    },
    {
        "name": "AP News",
        "type": "rss",
        "url": "https://feeds.apnews.com/rss/apf-topnews",
        "api_key_required": False,
        "rate_limit": None,
        "is_active": True
    }
]

# Fake news detection keywords (for demo purposes)
FAKE_NEWS_INDICATORS = {
    "suspicious_keywords": [
        "shocking", "unbelievable", "doctors hate", "secret", "miracle",
        "banned", "they don't want you to know", "incredible", "amazing",
        "breaking news", "urgent", "must read", "exclusive", "revealed"
    ],
    "clickbait_patterns": [
        "you won't believe",
        "what happens next will shock you",
        "number 7 will amaze you",
        "this one trick",
        "doctors hate him"
    ],
    "unreliable_domains": [
        "fakemews.com",
        "unreliablenews.net",
        "clickbaitcentral.org"
    ]
}

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service"""
    if service.lower() == "newsapi":
        return settings.newsapi_key
    elif service.lower() == "twitter":
        return settings.twitter_bearer_token
    return None

def is_source_active(source_name: str) -> bool:
    """Check if a data source should be active based on API key availability"""
    for source in DEFAULT_SOURCES:
        if source["name"] == source_name:
            if source["api_key_required"]:
                service = source_name.split()[0].lower()
                return get_api_key(service) is not None
            return source["is_active"]
    return False 