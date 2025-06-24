from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path
from app.config import settings
from pydantic_settings import BaseSettings

# Data models
class NewsArticle(BaseModel):
    id: str
    title: str
    content: str
    url: HttpUrl
    source: str
    published_date: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    language: str = "en"
    scraped_at: datetime

class DataSource(BaseModel):
    name: str
    type: str  # 'api', 'rss', 'social'
    url: HttpUrl
    api_key: Optional[str] = None
    is_active: bool = True
    last_scraped: Optional[datetime] = None

class PredictionRequest(BaseModel):
    text: str
    url: Optional[HttpUrl] = None
    source: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str  # 'real' or 'fake'
    confidence: float
    explanation: Optional[Dict[str, Any]] = None
    processed_at: datetime

# Initialize FastAPI app
app = FastAPI(
    title="Adaptive Fake News Detector - Data Gathering API",
    description="FastAPI backend for collecting and analyzing news data from multiple sources",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
articles_db: List[NewsArticle] = []
data_sources_db: List[DataSource] = []

# Utility functions
def save_article(article: NewsArticle):
    articles_db.append(article)
    # Persist all articles to a JSON file for later retrieval
    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump([a.dict() for a in articles_db], f, default=str, ensure_ascii=False, indent=2)
    print(f"Saved article: {article.title[:50]}...")

async def fetch_from_api(source: DataSource) -> List[NewsArticle]:
    """Fetch articles from news API"""
    articles = []
    try:
        async with httpx.AsyncClient() as client:
            headers = {}
            if source.api_key:
                headers["Authorization"] = f"Bearer {source.api_key}"
            
            response = await client.get(str(source.url), headers=headers, timeout=30.0)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response based on common API formats
            if "articles" in data:  # NewsAPI format
                for item in data["articles"]:
                    article = NewsArticle(
                        id=str(uuid.uuid4()),
                        title=item.get("title", ""),
                        content=item.get("content", "") or item.get("description", ""),
                        url=item.get("url"),
                        source=source.name,
                        published_date=datetime.fromisoformat(item.get("publishedAt", datetime.now().isoformat()).replace("Z", "+00:00")),
                        author=item.get("author"),
                        scraped_at=datetime.now()
                    )
                    articles.append(article)
            
    except Exception as e:
        print(f"Error fetching from {source.name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching from {source.name}: {str(e)}")
    
    return articles

async def fetch_from_rss(source: DataSource) -> List[NewsArticle]:
    """Fetch articles from RSS feed"""
    # Note: You'll need to add feedparser to requirements.txt
    articles = []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(str(source.url), timeout=30.0)
            response.raise_for_status()
            
            # Simple RSS parsing (in production, use feedparser)
            # For now, return empty list - implement with feedparser
            print(f"RSS parsing not implemented yet for {source.name}")
            
    except Exception as e:
        print(f"Error fetching RSS from {source.name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching RSS from {source.name}: {str(e)}")
    
    return articles

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Adaptive Fake News Detector API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "sources": "/sources",
            "articles": "/articles",
            "predict": "/predict",
            "gather": "/gather"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(),
        "articles_count": len(articles_db),
        "sources_count": len(data_sources_db)
    }

# Data Source Management
@app.get("/sources", response_model=List[DataSource])
async def get_data_sources():
    """Get all configured data sources"""
    return data_sources_db

@app.post("/sources", response_model=DataSource)
async def add_data_source(source: DataSource):
    """Add a new data source"""
    # Check if source already exists
    for existing_source in data_sources_db:
        if existing_source.url == source.url:
            raise HTTPException(status_code=400, detail="Data source with this URL already exists")
    
    data_sources_db.append(source)
    return source

@app.delete("/sources/{source_name}")
async def remove_data_source(source_name: str):
    """Remove a data source"""
    for i, source in enumerate(data_sources_db):
        if source.name == source_name:
            del data_sources_db[i]
            return {"message": f"Data source '{source_name}' removed successfully"}
    
    raise HTTPException(status_code=404, detail="Data source not found")

# Article Management
@app.get("/articles", response_model=List[NewsArticle])
async def get_articles(
    limit: int = 100,
    source: Optional[str] = None,
    since: Optional[datetime] = None
):
    """Get collected articles with optional filtering"""
    filtered_articles = articles_db
    
    if source:
        filtered_articles = [a for a in filtered_articles if a.source == source]
    
    if since:
        filtered_articles = [a for a in filtered_articles if a.scraped_at >= since]
    
    return filtered_articles[:limit]

@app.get("/articles/count")
async def get_articles_count():
    """Get count of articles by source"""
    counts = {}
    for article in articles_db:
        counts[article.source] = counts.get(article.source, 0) + 1
    
    return {
        "total": len(articles_db),
        "by_source": counts,
        "last_updated": max([a.scraped_at for a in articles_db]) if articles_db else None
    }

# Data Gathering
@app.post("/gather")
async def gather_data(background_tasks: BackgroundTasks, sources: Optional[List[str]] = None):
    """Trigger data gathering from configured sources"""
    if not data_sources_db:
        raise HTTPException(status_code=400, detail="No data sources configured")
    
    sources_to_process = data_sources_db
    if sources:
        sources_to_process = [s for s in data_sources_db if s.name in sources]
    
    if not sources_to_process:
        raise HTTPException(status_code=400, detail="No matching data sources found")
    
    # Add background task to gather data
    for source in sources_to_process:
        background_tasks.add_task(gather_from_source, source)
    
    return {
        "message": f"Data gathering started for {len(sources_to_process)} sources",
        "sources": [s.name for s in sources_to_process],
        "started_at": datetime.now()
    }

async def gather_from_source(source: DataSource):
    """Background task to gather data from a single source"""
    try:
        print(f"Starting data gathering from {source.name}...")
        
        if source.type == "api":
            articles = await fetch_from_api(source)
        elif source.type == "rss":
            articles = await fetch_from_rss(source)
        else:
            print(f"Unknown source type: {source.type}")
            return
        
        # Save articles
        for article in articles:
            save_article(article)
        
        # Update last scraped time
        source.last_scraped = datetime.now()
        
        print(f"Gathered {len(articles)} articles from {source.name}")
        
    except Exception as e:
        print(f"Error gathering from {source.name}: {str(e)}")

@app.post("/gather/single")
async def gather_from_single_source(source_name: str, background_tasks: BackgroundTasks):
    """Gather data from a single source immediately"""
    source = next((s for s in data_sources_db if s.name == source_name), None)
    if not source:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    background_tasks.add_task(gather_from_source, source)
    
    return {
        "message": f"Data gathering started for {source_name}",
        "started_at": datetime.now()
    }

# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: PredictionRequest):
    """Predict if news text is fake or real"""
    # Placeholder prediction logic - replace with actual ML model
    import random
    
    # Simple rule-based prediction for demonstration
    fake_keywords = ["shocking", "unbelievable", "doctors hate", "secret", "miracle"]
    text_lower = request.text.lower()
    
    fake_score = sum(1 for keyword in fake_keywords if keyword in text_lower)
    confidence = min(0.9, 0.5 + (fake_score * 0.1))
    prediction = "fake" if fake_score > 0 else "real"
    
    # Add some randomness for demonstration
    if random.random() < 0.3:  # 30% chance to flip
        prediction = "fake" if prediction == "real" else "real"
        confidence = max(0.5, random.uniform(0.6, 0.9))
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        explanation={
            "keywords_found": fake_keywords if fake_score > 0 else [],
            "text_length": len(request.text),
            "source": request.source
        },
        processed_at=datetime.now()
    )

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Predict multiple news texts"""
    results = []
    for req in requests:
        prediction = await predict_news(req)
        results.append(prediction)
    
    return {
        "predictions": results,
        "total_processed": len(results),
        "processed_at": datetime.now()
    }

# Analytics endpoints
@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary"""
    if not articles_db:
        return {"message": "No articles collected yet"}
    
    # Calculate statistics
    total_articles = len(articles_db)
    sources = list(set(a.source for a in articles_db))
    date_range = {
        "earliest": min(a.published_date for a in articles_db),
        "latest": max(a.published_date for a in articles_db)
    }
    
    # Articles by day for the last 7 days
    recent_articles = [a for a in articles_db if a.scraped_at >= datetime.now() - timedelta(days=7)]
    daily_counts = {}
    for article in recent_articles:
        day = article.scraped_at.strftime("%Y-%m-%d")
        daily_counts[day] = daily_counts.get(day, 0) + 1
    
    return {
        "total_articles": total_articles,
        "total_sources": len(sources),
        "sources": sources,
        "date_range": date_range,
        "last_7_days": daily_counts,
        "average_per_day": len(recent_articles) / 7 if recent_articles else 0
    }

# Initialize with some sample data sources
@app.on_event("startup")
async def startup_event():
    """Initialize the application with sample data sources"""
    sample_sources = [
        DataSource(
            name="NewsAPI General",
            type="api",
            url="https://newsapi.org/v2/top-headlines?country=us&category=general",
            api_key=settings.newsapi_key,
            is_active=bool(settings.newsapi_key)
        ),
        DataSource(
            name="BBC RSS",
            type="rss",
            url="http://feeds.bbci.co.uk/news/rss.xml",
            is_active=True
        ),
        DataSource(
            name="CNN RSS",
            type="rss", 
            url="http://rss.cnn.com/rss/edition.rss",
            is_active=True
        )
    ]
    
    data_sources_db.extend(sample_sources)
    print(f"Initialized with {len(sample_sources)} sample data sources")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
