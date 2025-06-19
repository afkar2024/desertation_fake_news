"""
Data processing utilities for the Fake News Detector
"""
import re
import feedparser
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import uuid

from .config import settings, FAKE_NEWS_INDICATORS

async def fetch_rss_articles(rss_url: str, max_articles: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch and parse articles from RSS feed
    """
    articles = []
    
    try:
        # Fetch RSS content
        async with httpx.AsyncClient() as client:
            response = await client.get(rss_url, timeout=30.0)
            response.raise_for_status()
            
        # Parse RSS feed
        feed = feedparser.parse(response.content)
        
        if feed.bozo:
            print(f"Warning: RSS feed might be malformed: {rss_url}")
        
        # Extract articles
        for entry in feed.entries[:max_articles]:
            article = {
                "id": str(uuid.uuid4()),
                "title": entry.get("title", "").strip(),
                "content": extract_content_from_entry(entry),
                "url": entry.get("link", ""),
                "source": feed.feed.get("title", urlparse(rss_url).netloc),
                "published_date": parse_date(entry),
                "author": entry.get("author", None),
                "category": extract_category(entry),
                "language": "en",  # Default to English
                "scraped_at": datetime.now(timezone.utc)
            }
            
            # Skip empty articles
            if article["title"] and article["url"]:
                articles.append(article)
                
    except Exception as e:
        print(f"Error fetching RSS from {rss_url}: {str(e)}")
        raise
    
    return articles

def extract_content_from_entry(entry) -> str:
    """Extract content from RSS entry"""
    content = ""
    
    # Try different content fields
    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].value if isinstance(entry.content, list) else entry.content
    elif hasattr(entry, 'summary'):
        content = entry.summary
    elif hasattr(entry, 'description'):
        content = entry.description
    
    # Clean HTML tags
    if content:
        content = clean_html(content)
    
    return content.strip()

def extract_category(entry) -> Optional[str]:
    """Extract category from RSS entry"""
    if hasattr(entry, 'tags') and entry.tags:
        return entry.tags[0].term if entry.tags else None
    elif hasattr(entry, 'category'):
        return entry.category
    return None

def parse_date(entry) -> datetime:
    """Parse publication date from RSS entry"""
    try:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        elif hasattr(entry, 'published'):
            # Try to parse string date
            from dateutil import parser
            return parser.parse(entry.published)
    except Exception as e:
        print(f"Error parsing date: {e}")
    
    # Default to current time if parsing fails
    return datetime.now(timezone.utc)

def clean_html(text: str) -> str:
    """Remove HTML tags and clean text"""
    if not text:
        return ""
    
    # Parse HTML
    soup = BeautifulSoup(text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text and clean whitespace
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ""

def is_suspicious_domain(url: str) -> bool:
    """Check if domain is in suspicious list"""
    domain = extract_domain(url)
    return domain in FAKE_NEWS_INDICATORS["unreliable_domains"]

def calculate_fake_news_score(title: str, content: str, url: str = "") -> Dict[str, Any]:
    """
    Calculate fake news probability based on simple heuristics
    This is a placeholder - replace with actual ML model
    """
    score = 0.0
    indicators = []
    
    text = f"{title} {content}".lower()
    
    # Check suspicious keywords
    suspicious_count = 0
    for keyword in FAKE_NEWS_INDICATORS["suspicious_keywords"]:
        if keyword in text:
            suspicious_count += 1
            indicators.append(f"Contains suspicious keyword: '{keyword}'")
    
    # Check clickbait patterns
    clickbait_count = 0
    for pattern in FAKE_NEWS_INDICATORS["clickbait_patterns"]:
        if pattern in text:
            clickbait_count += 1
            indicators.append(f"Contains clickbait pattern: '{pattern}'")
    
    # Check domain reputation
    domain_suspicious = is_suspicious_domain(url)
    if domain_suspicious:
        indicators.append(f"Suspicious domain: {extract_domain(url)}")
    
    # Calculate score (0-1, where 1 is most likely fake)
    score = min(1.0, (suspicious_count * 0.1) + (clickbait_count * 0.2) + (0.3 if domain_suspicious else 0))
    
    # Add some content-based heuristics
    if len(title) > 100:  # Very long titles are often clickbait
        score += 0.1
        indicators.append("Unusually long title")
    
    if text.count('!') > 3:  # Excessive exclamation marks
        score += 0.1
        indicators.append("Excessive exclamation marks")
    
    if re.search(r'\b[A-Z]{4,}\b', title):  # ALL CAPS words in title
        score += 0.1
        indicators.append("ALL CAPS words in title")
    
    return {
        "fake_probability": min(1.0, score),
        "confidence": min(0.9, 0.5 + score),
        "indicators": indicators,
        "suspicious_keywords": suspicious_count,
        "clickbait_patterns": clickbait_count,
        "domain_suspicious": domain_suspicious
    }

async def fetch_article_content(url: str) -> Optional[str]:
    """
    Fetch full article content from URL
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            
            # Parse HTML and extract main content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Try to find main content
            content_selectors = [
                'article',
                '.article-content',
                '.content',
                '.post-content',
                '.entry-content',
                'main',
                '#content'
            ]
            
            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(strip=True)
                    break
            
            # Fallback to body content
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(strip=True)
            
            return content[:5000] if content else None  # Limit content length
            
    except Exception as e:
        print(f"Error fetching article content from {url}: {e}")
        return None

def validate_article_data(article: Dict[str, Any]) -> bool:
    """
    Validate article data completeness and quality
    """
    required_fields = ['title', 'url', 'source']
    
    # Check required fields
    for field in required_fields:
        if not article.get(field):
            return False
    
    # Basic quality checks
    if len(article['title']) < 10:  # Title too short
        return False
        
    if not article['url'].startswith(('http://', 'https://')):
        return False
    
    return True

def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate articles based on URL and title similarity
    """
    seen_urls = set()
    seen_titles = set()
    unique_articles = []
    
    for article in articles:
        url = article.get('url', '').lower()
        title = article.get('title', '').lower().strip()
        
        # Skip if URL already seen
        if url in seen_urls:
            continue
            
        # Skip if very similar title already seen
        title_words = set(title.split())
        is_duplicate = False
        
        for seen_title in seen_titles:
            seen_words = set(seen_title.split())
            if len(title_words.intersection(seen_words)) / max(len(title_words), len(seen_words)) > 0.8:
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_urls.add(url)
            seen_titles.add(title)
            unique_articles.append(article)
    
    return unique_articles

def get_article_statistics(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics for a collection of articles
    """
    if not articles:
        return {"total": 0}
    
    sources = {}
    categories = {}
    dates = []
    
    for article in articles:
        # Count by source
        source = article.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
        
        # Count by category
        category = article.get('category', 'Uncategorized')
        categories[category] = categories.get(category, 0) + 1
        
        # Collect dates
        if 'published_date' in article:
            dates.append(article['published_date'])
    
    stats = {
        "total": len(articles),
        "sources": sources,
        "categories": categories,
        "date_range": {
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None
        },
        "avg_title_length": sum(len(a.get('title', '')) for a in articles) / len(articles),
        "avg_content_length": sum(len(a.get('content', '')) for a in articles) / len(articles)
    }
    
    return stats 