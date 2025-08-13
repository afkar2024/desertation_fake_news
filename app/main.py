from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
import traceback
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from datetime import datetime, timedelta
import json
import uuid
import warnings
from pathlib import Path
from contextlib import asynccontextmanager
from app.config import settings
from app.data_utils import fetch_rss_articles, fetch_article_content
from pydantic_settings import BaseSettings
from collections import deque
from scipy.stats import ks_2samp
from app.dataset_api import router as dataset_router
from app.model_service import model_service
from app.feedback_api import router as feedback_router
from app.eval_progress import progress_manager
from fastapi import WebSocket, WebSocketDisconnect

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="textstat")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="resume_download is deprecated")

# Data models
class NewsArticle(BaseModel):
    id: str
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    language: str = "en"
    scraped_at: datetime

class DataSource(BaseModel):
    name: str
    type: str  # 'api', 'rss', 'social'
    url: str
    api_key: Optional[str] = None
    is_active: bool = True
    last_scraped: Optional[datetime] = None

class PredictionRequest(BaseModel):
    text: str
    url: Optional[str] = None
    source: Optional[str] = None
    abstain_threshold: Optional[float] = None  # abstain if uncertainty metric below/above threshold
    abstain_metric: Optional[str] = "margin"   # 'margin' (abstain if margin < thr) or 'entropy' (abstain if entropy > thr)

class PredictionResponse(BaseModel):
    prediction: str  # 'real' or 'fake'
    confidence: float
    explanation: Optional[Dict[str, Any]] = None
    processed_at: datetime

class ExplanationRequest(BaseModel):
    text: str
    max_evals: Optional[int] = None

class PredictUrlRequest(BaseModel):
    url: str
    max_chars: int = 5000
    source: Optional[str] = None

class PredictExplainRequest(BaseModel):
    text: str
    max_evals: Optional[int] = None
    top_tokens: int = 30
    use_lime: bool = False
    use_attention: bool = False
    explanation_confidence: bool = False

# In-memory storage (replace with database in production)
articles_db: List[NewsArticle] = []
data_sources_db: List[DataSource] = []
prediction_log: List[Dict[str, Any]] = []  # {'ts': datetime, 'prob_fake': float, 'prob_real': float}

# Drift monitoring (lightweight):
recent_fake_probs: deque[float] = deque(maxlen=2000)
baseline_fake_probs: List[float] = []
baseline_set_at: Optional[str] = None
KS_P_THRESHOLD: float = 0.01

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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
    
    yield
    
    # Shutdown (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Adaptive Fake News Detector - Data Gathering API",
    description="FastAPI backend for collecting and analyzing news data from multiple sources",
    version="1.0.0",
    lifespan=lifespan
)

# Global exception handler with detailed context
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    tb = traceback.TracebackException.from_exception(exc)
    last = None
    for frame in tb.stack:
        last = frame
    location = {
        "file": getattr(last, 'filename', None),
        "function": getattr(last, 'name', None),
        "line": getattr(last, 'lineno', None),
        "url": str(request.url) if hasattr(request, 'url') else None,
        "method": getattr(request, 'method', None),
    }
    print("[EXCEPTION]", {
        "type": type(exc).__name__,
        "message": str(exc),
        "location": location,
    })
    return JSONResponse(status_code=500, content={
        "detail": f"Internal Server Error: {type(exc).__name__}",
        "error": str(exc),
        "location": location,
    })

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include dataset router
app.include_router(dataset_router)
app.include_router(feedback_router)
# WebSocket for evaluation progress
@app.websocket("/ws/evaluation/{trace_id}")
async def ws_evaluation(websocket: WebSocket, trace_id: str):
    await websocket.accept()
    progress_manager.attach(trace_id, websocket)
    try:
        # If there are queued messages, they will be pushed by publisher
        # Send hello
        await websocket.send_json({"type": "hello", "trace_id": trace_id})
        # Keep open until client disconnects
        while True:
            # Simple ping to keep alive
            await asyncio.sleep(10)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        progress_manager.detach(trace_id, websocket)


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
    articles: List[NewsArticle] = []
    try:
        raw_articles = await fetch_rss_articles(str(source.url), max_articles=settings.max_articles_per_source)
        for item in raw_articles:
            try:
                article = NewsArticle(
                    id=item.get("id", str(uuid.uuid4())),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    url=item.get("url", ""),
                    source=item.get("source", source.name),
                    published_date=item.get("published_date") or datetime.now(),
                    author=item.get("author"),
                    category=item.get("category"),
                    language=item.get("language", "en"),
                    scraped_at=item.get("scraped_at") or datetime.now(),
                )
                articles.append(article)
            except Exception as inner_e:
                print(f"Skipping malformed RSS article from {source.name}: {inner_e}")
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
@app.get("/gather")
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
    """Predict if news text is fake or real using transformer model"""
    result = model_service.classify_text(request.text)
    pred_label = "fake" if result["prediction"] == 1 else "real"
    abstained: Optional[bool] = None
    if request.abstain_threshold is not None:
        if (request.abstain_metric or "margin").lower() == "entropy":
            ent = float(result.get("uncertainty", {}).get("predictive_entropy", 0.0))
            abstained = ent > float(request.abstain_threshold)
        else:
            margin = float(result.get("uncertainty", {}).get("margin", 0.0))
            abstained = margin < float(request.abstain_threshold)
        if abstained:
            pred_label = "abstain"
    explanation = {
        "text_length": len(request.text or ""),
        "source": request.source,
        "probabilities": result.get("probabilities", {}),
    }
    # Log prediction for monitoring
    try:
        p = result.get("probabilities", {})
        fake_p = float(p.get("fake", 0.0))
        prediction_log.append({
            "ts": datetime.now().isoformat(),
            "prob_fake": fake_p,
            "prob_real": float(p.get("real", 0.0)),
        })
        # cap log size
        if len(prediction_log) > 5000:
            del prediction_log[: len(prediction_log) - 5000]
        # update drift buffers
        recent_fake_probs.append(fake_p)
    except Exception:
        pass
    return PredictionResponse(
        prediction=pred_label,
        confidence=float(result["confidence"]),
        explanation=explanation,
        processed_at=datetime.now(),
        **({"abstained": abstained} if abstained is not None else {}),
    )

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Predict multiple news texts using transformer model"""
    texts = [r.text for r in requests]
    batch_results = model_service.classify_batch(texts)
    responses: List[PredictionResponse] = []
    for i, r in enumerate(batch_results):
        pred_label = "fake" if r["prediction"] == 1 else "real"
        abstained: Optional[bool] = None
        req = requests[i]
        if req.abstain_threshold is not None:
            metric = (req.abstain_metric or "margin").lower()
            if metric == "entropy":
                ent = float(r.get("uncertainty", {}).get("predictive_entropy", 0.0))
                abstained = ent > float(req.abstain_threshold)
            else:
                margin = float(r.get("uncertainty", {}).get("margin", 0.0))
                abstained = margin < float(req.abstain_threshold)
            if abstained:
                pred_label = "abstain"
        responses.append(
            PredictionResponse(
                prediction=pred_label,
                confidence=float(r["confidence"]),
                explanation={
                    "text_length": len(texts[i] or ""),
                    "source": req.source,
                    "probabilities": r.get("probabilities", {}),
                },
                processed_at=datetime.now(),
                **({"abstained": abstained} if abstained is not None else {}),
            )
        )
    return {
        "predictions": responses,
        "total_processed": len(responses),
        "processed_at": datetime.now(),
    }

# Model management endpoints
@app.get("/model/info")
async def get_model_info():
    return model_service.get_model_info()


class ModelReloadRequest(BaseModel):
    model_source: Optional[str] = None  # path or HF model name
    temperature: Optional[float] = None


@app.post("/model/reload")
async def reload_model(request: ModelReloadRequest):
    try:
        info = model_service.reload(request.model_source)
        if request.temperature is not None:
            model_service.set_temperature(float(request.temperature))
            info = model_service.get_model_info()
        return {"message": "Model reloaded", "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


class TemperatureCalibrateRequest(BaseModel):
    texts: List[str]
    labels: List[int]
    max_steps: int = 200
    lr: float = 0.01


@app.post("/model/calibrate-temperature")
async def calibrate_temperature(req: TemperatureCalibrateRequest):
    """Calibrate temperature on provided labeled texts (logit scaling)."""
    try:
        if not req.texts or not req.labels or len(req.texts) != len(req.labels):
            raise HTTPException(status_code=400, detail="texts and labels must be non-empty and have same length")
        new_t = model_service.calibrate_temperature(req.texts, req.labels, max_steps=req.max_steps, lr=req.lr)
        return {"temperature": new_t}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")

# Explanation endpoint (SHAP)
@app.post("/explain/shap")
async def explain_prediction(request: ExplanationRequest):
    """Generate SHAP explanation for a single text"""
    if not settings.enable_explanations:
        raise HTTPException(status_code=400, detail="Explanations are disabled")

    try:
        exp = model_service.explain_text(request.text, max_evals=request.max_evals)
        return {
            "tokens": exp["tokens"],
            "shap_values": exp["shap_values"],
            "base_value": exp["base_value"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.post("/predict/url", response_model=PredictionResponse)
async def predict_from_url(request: PredictUrlRequest):
    """Fetch article content from URL and predict using transformer model"""
    try:
        content = await fetch_article_content(request.url)
        if not content:
            raise HTTPException(status_code=400, detail="No content extracted from URL")
        text = content[: max(1, request.max_chars)]
        result = model_service.classify_text(text)
        pred_label = "fake" if result["prediction"] == 1 else "real"
        return PredictionResponse(
            prediction=pred_label,
            confidence=float(result["confidence"]),
            explanation={
                "url": request.url,
                "text_length": len(text),
                "probabilities": result.get("probabilities", {}),
                "source": request.source,
            },
            processed_at=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL prediction error: {str(e)}")


@app.post("/predict/explain")
async def predict_with_explanation(request: PredictExplainRequest):
    """Return prediction and SHAP explanation together for a single text"""
    try:
        pred = model_service.classify_text(request.text)
        if request.explanation_confidence:
            ec = model_service.explanation_confidence(request.text, top_k=max(1, min(request.top_tokens, 10)))
            exp_preview = {"explanation_confidence": ec}
        elif request.use_attention:
            exp = model_service.explain_text_attention(request.text)
            if not isinstance(exp, dict):
                exp = {"tokens": [], "weights": []}
            # return top tokens by weight
            tokens = exp.get("tokens", []) if isinstance(exp, dict) else []
            weights = exp.get("weights", []) if isinstance(exp, dict) else []
            # pair and sort
            pairs = list(zip(tokens, weights))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[: max(1, request.top_tokens)]
            exp_preview = {
                "attention_tokens": [p[0] for p in pairs_sorted],
                "attention_weights": [float(p[1]) for p in pairs_sorted],
            }
        elif request.use_lime:
            exp = model_service.explain_text_lime(request.text, num_features=max(1, request.top_tokens))
            if not isinstance(exp, dict):
                exp = {"features": [], "weights": []}
            exp_preview = {"features": exp.get("features", [])[: max(1, request.top_tokens)], "weights": exp.get("weights", [])[: max(1, request.top_tokens)]}
        else:
            exp = model_service.explain_text(request.text, max_evals=request.max_evals)
            # Optionally truncate tokens for response brevity
            top_n = max(1, request.top_tokens)
            exp_preview = {
                "tokens": exp.get("tokens", [])[:top_n],
                "shap_values": exp.get("shap_values", [])[:top_n],
                "base_value": exp.get("base_value"),
            }
        return {
            "prediction": "fake" if pred["prediction"] == 1 else "real",
            "confidence": float(pred["confidence"]),
            "probabilities": pred.get("probabilities", {}),
            "explanation": exp_preview,
            "processed_at": datetime.now(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict+Explain error: {str(e)}")


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Serve a minimal HTML demo page for prediction + SHAP preview."""
    return """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Fake News Demo</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 24px; max-width: 900px; }
    textarea { width: 100%; height: 160px; font-size: 14px; }
    .row { margin: 12px 0; }
    .btn { padding: 8px 14px; border: 1px solid #ccc; background: #f7f7f7; cursor: pointer; }
    .chip { display: inline-block; padding: 4px 10px; border-radius: 12px; margin-right: 8px; }
    .real { background: #e8f5e9; color: #2e7d32; }
    .fake { background: #ffebee; color: #c62828; }
    .token { display: inline-block; margin: 2px; padding: 2px 4px; border-radius: 4px; font-family: monospace; }
  </style>
  </head>
  <body>
    <h1>Fake News Detection - Demo</h1>
    <div class=\"row\">
      <label for=\"text\">Enter text to analyze:</label>
      <textarea id=\"text\" placeholder=\"Paste news text here...\"></textarea>
    </div>
    <div class=\"row\">
      <button class=\"btn\" onclick=\"predictExplain()\">Predict + Explain</button>
      <span id=\"status\"></span>
    </div>
    <div class=\"row\">
      <div id=\"result\"></div>
    </div>
    <script>
      async function predictExplain() {
        const text = document.getElementById('text').value;
        const status = document.getElementById('status');
        const resultEl = document.getElementById('result');
        resultEl.innerHTML = '';
        status.textContent = 'Processing...';
        try {
          const resp = await fetch('/predict/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, top_tokens: 40 })
          });
          if (!resp.ok) {
            const errTxt = await resp.text();
            throw new Error(errTxt);
          }
          const data = await resp.json();
          const pred = data.prediction;
          const conf = (data.confidence * 100).toFixed(1);
          const probs = data.probabilities || {};
          const exp = data.explanation || {};
          const tokens = exp.tokens || [];
          const shap = exp.shap_values || [];

          const chip = document.createElement('span');
          chip.className = 'chip ' + (pred === 'fake' ? 'fake' : 'real');
          chip.textContent = pred === 'fake' ? `Fake News (${conf}%)` : `Real News (${conf}%)`;
          resultEl.appendChild(chip);

          const p = document.createElement('div');
          p.textContent = `P(real)=${(probs.real*100||0).toFixed(1)}%, P(fake)=${(probs.fake*100||0).toFixed(1)}%`;
          resultEl.appendChild(p);

          const expl = document.createElement('div');
          expl.style.marginTop = '10px';
          expl.innerHTML = '<strong>Explanation (top tokens):</strong><br/>';
          for (let i=0; i<tokens.length; i++) {
            const t = document.createElement('span');
            t.className = 'token';
            const val = shap[i] || 0;
            const abs = Math.min(Math.abs(val), 1.0);
            const bg = val > 0 ? `rgba(244,67,54,${0.15 + abs*0.6})` : `rgba(76,175,80,${0.15 + abs*0.6})`;
            t.style.background = bg;
            t.textContent = tokens[i];
            expl.appendChild(t);
          }
          resultEl.appendChild(expl);
          status.textContent = 'Done.';
        } catch (e) {
          status.textContent = 'Error: ' + e.message;
        }
      }
    </script>
  </body>
  </html>
    """


class CounterfactualRequest(BaseModel):
    text: str
    max_candidates: int = 3


@app.post("/predict/counterfactual")
async def generate_counterfactuals(req: CounterfactualRequest):
    """Generate simple counterfactuals by removing top SHAP tokens and re-predicting."""
    try:
        variants = model_service.generate_counterfactuals(req.text, max_candidates=max(1, req.max_candidates))
        return {"base_text": req.text, "counterfactuals": variants}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counterfactual generation error: {str(e)}")

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
    
    payload = {
        "total_articles": total_articles,
        "total_sources": len(sources),
        "sources": sources,
        "date_range": date_range,
        "last_7_days": daily_counts,
        "average_per_day": len(recent_articles) / 7 if recent_articles else 0,
        "prediction_monitor": {
            "count": len(prediction_log),
            "avg_prob_fake": (sum(x.get("prob_fake", 0.0) for x in prediction_log) / len(prediction_log)) if prediction_log else 0.0,
            "avg_prob_real": (sum(x.get("prob_real", 0.0) for x in prediction_log) / len(prediction_log)) if prediction_log else 0.0,
        }
    }

    # Lightweight concept drift check using KS test on P(fake)
    try:
        global baseline_fake_probs, baseline_set_at
        if not baseline_fake_probs and len(recent_fake_probs) >= 500:
            baseline_fake_probs = list(recent_fake_probs)
            baseline_set_at = datetime.now().isoformat()
        elif baseline_fake_probs and len(recent_fake_probs) >= 200:
            stat, pval = ks_2samp(baseline_fake_probs, list(recent_fake_probs))
            payload["prediction_monitor"]["drift"] = {
                "ks_stat": float(stat),
                "p_value": float(pval),
                "is_drift": bool(pval < KS_P_THRESHOLD),
                "baseline_set_at": baseline_set_at,
                "baseline_size": len(baseline_fake_probs),
                "recent_size": len(recent_fake_probs),
            }
    except Exception:
        pass

    # Provide top-level aliases for frontend convenience
    try:
        pm = payload.get("prediction_monitor", {})
        payload["prediction_count"] = int(pm.get("count", 0))
        payload["avg_prob_fake"] = float(pm.get("avg_prob_fake", 0.0))
        payload["avg_prob_real"] = float(pm.get("avg_prob_real", 0.0))
        drift = (pm.get("drift") or {})
        if "p_value" in drift:
            payload["drift_ks_p_value"] = float(drift.get("p_value"))
        # Optional: include short histories if available in log
        # Compute recent rolling averages over last 50 predictions
        if prediction_log:
            window = min(50, len(prediction_log))
            recent = prediction_log[-window:]
            # simple per-sample history of probs (not rolling average), for sparkline
            payload["avg_prob_fake_history"] = [float(x.get("prob_fake", 0.0)) for x in recent]
            payload["avg_prob_real_history"] = [float(x.get("prob_real", 0.0)) for x in recent]
    except Exception:
        pass

    # Attach latest evaluation report summary, if any
    try:
        from app.cache_store import list_reports_json, get_report_json
        items = list_reports_json(limit=1)
        latest = None
        for it in items:
            # find most recent evaluation report
            if it.get("report_type") == "evaluation":
                latest = it
                break
        if latest:
            rep = get_report_json(int(latest.get("id")))
            payload["last_evaluation"] = {
                "dataset": rep.get("dataset"),
                "created_at": rep.get("created_at"),
                "metrics": {
                    "accuracy": rep.get("payload", {}).get("accuracy"),
                    "precision": rep.get("payload", {}).get("precision"),
                    "recall": rep.get("payload", {}).get("recall"),
                    "f1": rep.get("payload", {}).get("f1"),
                },
                "size": rep.get("payload", {}).get("total_evaluated"),
            }
    except Exception:
        pass

    return payload

# Reports browsing endpoints (frontend markdown preview)
@app.get("/reports")
async def list_reports():
    try:
        from app.cache_store import list_reports_json
        items = list_reports_json(limit=50)
        return {"items": items, "format": "json"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@app.get("/reports/{report_id}")
async def get_report(report_id: int):
    try:
        from app.cache_store import get_report_json
        item = get_report_json(report_id)
        if not item:
            raise HTTPException(status_code=404, detail="Report not found")
        return item
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
