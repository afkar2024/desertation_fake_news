# Adaptive Fake News Detector

A comprehensive FastAPI backend for real-time news data collection and fake news detection from multiple sources.

## Features

- **Multi-Source Data Gathering**: Collect news from APIs, RSS feeds, and social media
- **Real-time Prediction**: Instant fake news detection with confidence scores
- **Adaptive Retraining**: Weekly model updates on user-flagged data  
- **Explainability**: Detailed analysis with reasoning behind predictions
- **RESTful API**: Complete REST API with automatic documentation
- **Background Processing**: Asynchronous data collection and processing
- **Analytics Dashboard**: Track data collection and prediction statistics

## Project Structure

```
desertation_fake_news/
├── app/
│   ├── __init__.py
│   ├── main.py          # Main FastAPI application
│   └── config.py        # Configuration settings
├── requirements.txt     # Python dependencies
├── start_server.py     # Server startup script
├── env.example         # Environment configuration template
└── README.md
```

## Quick Start

### Using uv (Recommended - Faster!)

1. **Install Dependencies**:
   ```bash
   # If you haven't created a virtual environment yet:
   uv venv
   
   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   # Install dependencies
   uv pip install -e .
   ```

2. **Configure Environment** (Optional):
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Start the Server**:
   ```bash
   python start_server.py
   ```

   Or directly with uvicorn:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Using pip (Traditional)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment** (Optional):
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Start the Server**:
   ```bash
   python start_server.py
   ```

4. **Access the API**:
   - **Main API**: http://localhost:8000
   - **Interactive Docs**: http://localhost:8000/docs
   - **ReDoc Documentation**: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Health check with system status
- `POST /predict` - Predict if news text is fake or real
- `POST /predict/batch` - Batch prediction for multiple texts

### Data Source Management
- `GET /sources` - List all configured data sources
- `POST /sources` - Add a new data source
- `DELETE /sources/{source_name}` - Remove a data source

### Article Management
- `GET /articles` - Retrieve collected articles (with filtering)
- `GET /articles/count` - Get article statistics by source

### Data Collection
- `POST /gather` - Trigger data gathering from all sources
- `POST /gather/single` - Gather data from a specific source

### Analytics
- `GET /analytics/summary` - Get comprehensive analytics summary

## Example Usage

### 1. Add a Data Source
```bash
curl -X POST "http://localhost:8000/sources" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "TechCrunch RSS",
    "type": "rss",
    "url": "https://techcrunch.com/feed/",
    "is_active": true
  }'
```

### 2. Gather Data
```bash
curl -X POST "http://localhost:8000/gather"
```

### 3. Predict Fake News
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "SHOCKING: Scientists discover this ONE weird trick that doctors dont want you to know!",
    "source": "example.com"
  }'
```

### 4. Get Articles
```bash
curl "http://localhost:8000/articles?limit=10&source=BBC"
```

## Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# API Keys
NEWSAPI_KEY=your_newsapi_key_here
TWITTER_BEARER_TOKEN=your_twitter_token_here

# Server Settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Data Collection
MAX_ARTICLES_PER_SOURCE=100
COLLECTION_INTERVAL_HOURS=6
```

### Getting API Keys

1. **NewsAPI**: Register at [newsapi.org](https://newsapi.org/register)
2. **Twitter API**: Apply at [developer.twitter.com](https://developer.twitter.com/en/portal/dashboard)

## Default Data Sources

The application comes pre-configured with several RSS feeds:

- **BBC News** (World & Technology)
- **CNN RSS**
- **Reuters World News** 
- **AP News**
- **NewsAPI** (requires API key)

## Development

### Development Setup with uv

```bash
# Clone and setup the project
git clone <your-repo>
cd desertation_fake_news

# Create virtual environment
uv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install project in development mode with all dev dependencies
uv pip install -e ".[dev,ml]"

# Run code formatting
black .
ruff check .

# Run tests (when added)
pytest
```

### Why Use uv?

- ⚡ **10-100x faster** than pip for dependency resolution and installation
- 🔒 **Better dependency resolution** - resolves conflicts more reliably
- 📦 **Modern Python packaging** - full support for pyproject.toml
- 🎯 **Drop-in replacement** - works with existing pip workflows
- 🔄 **Cross-platform** - consistent behavior across Windows, macOS, Linux

### Development Commands with uv

```bash
# Add a new dependency
uv pip install requests

# Add development dependency
uv pip install pytest --dev

# Install from requirements.txt (if needed)
uv pip install -r requirements.txt

# Upgrade all packages
uv pip install --upgrade-strategy eager --upgrade .

# Export current environment
uv pip freeze > requirements-lock.txt
```

### Using Makefile (Recommended)

For easier development workflow, use the included Makefile:

```bash
# See all available commands
make help

# Install dependencies
make install          # Basic dependencies
make install-dev      # With dev dependencies  
make install-ml       # With ML dependencies

# Development tasks
make run              # Start the server
make format           # Format code with black
make lint             # Check code with ruff
make test             # Run tests
make clean            # Clean up temporary files
```

### Adding New Data Sources

1. **RSS Feeds**: Just add the RSS URL
2. **APIs**: Implement the parsing logic in `fetch_from_api()`
3. **Social Media**: Add authentication and API calls

### Extending Prediction Logic

The current prediction uses simple heuristics. To improve:

1. Replace `predict_news()` with ML model
2. Add training data collection
3. Implement SHAP explanations
4. Add user feedback loop

## Deployment

### Docker (Coming Soon)
```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

### Production Considerations

- Use a proper database (PostgreSQL/MongoDB)
- Add authentication and rate limiting
- Implement proper logging and monitoring
- Use Redis for background task queue
- Add caching for repeated requests

## Next Steps

Based on your pilot study requirements:

1. **Data Collection Enhancement**:
   - Add more news sources
   - Implement social media data collection
   - Add data validation and cleaning

2. **Machine Learning Integration**:
   - Train actual fake news detection models
   - Implement SHAP explanations
   - Add user feedback collection

3. **Database Integration**:
   - Replace in-memory storage with PostgreSQL
   - Add data persistence and backup

4. **Frontend Development**:
   - Build React dashboard
   - Add real-time updates
   - Create admin interface

## License

This project is part of a dissertation research on fake news detection.
