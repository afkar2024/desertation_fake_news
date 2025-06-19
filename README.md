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
├── Makefile           # Build automation (macOS/Linux)
├── make.bat           # Build automation (Windows)
├── env.example         # Environment configuration template
└── README.md
```

## Prerequisites

### Operating System Setup

#### Windows
- **Python**: Download from [python.org](https://www.python.org/downloads/) (3.8+)
- **Make** (Optional): Install via winget: `winget install GnuWin32.Make`
- **Alternative**: Use the included `make.bat` file (recommended for Windows)

#### macOS
- **Python**: Install via Homebrew: `brew install python` or download from [python.org](https://www.python.org/downloads/)
- **Make**: Usually pre-installed. If not: `xcode-select --install`
- **Homebrew** (if not installed): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv make
```

#### Linux (CentOS/RHEL/Fedora)
```bash
sudo yum install python3 python3-pip make
# or for newer versions:
sudo dnf install python3 python3-pip make
```

### Install uv (Recommended Package Manager)

uv is a fast Python package installer and resolver. Install it on any OS:

```bash
# Using pip
pip install uv

# Using curl (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using PowerShell (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
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

## Build Automation

### Using Makefile (macOS/Linux)

The project includes a Makefile for common development tasks:

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

### Using make.bat (Windows)

For Windows users, use the included batch file:

```cmd
# See all available commands
make.bat

# Install dependencies
make.bat install          # Basic dependencies
make.bat install-dev      # With dev dependencies  
make.bat install-ml       # With ML dependencies

# Development tasks
make.bat run              # Start the server
make.bat format           # Format code with black
make.bat lint             # Check code with ruff
make.bat test             # Run tests
make.bat clean            # Clean up temporary files
```

### Make Installation by OS

#### Windows
```cmd
# Using winget (recommended)
winget install GnuWin32.Make

# Using Chocolatey
choco install make

# Using the full path after installation
"C:\Program Files (x86)\GnuWin32\bin\make.exe" help
```

#### macOS
```bash
# Usually pre-installed, but if not:
xcode-select --install

# Or using Homebrew
brew install make
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install make
```

#### Linux (CentOS/RHEL/Fedora)
```bash
# CentOS/RHEL
sudo yum install make

# Fedora
sudo dnf install make
```

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

### Cross-Platform Development Workflow

#### Using Makefile (macOS/Linux)
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

#### Using make.bat (Windows)
```cmd
# See all available commands
make.bat

# Install dependencies
make.bat install          # Basic dependencies
make.bat install-dev      # With dev dependencies  
make.bat install-ml       # With ML dependencies

# Development tasks
make.bat run              # Start the server
make.bat format           # Format code with black
make.bat lint             # Check code with ruff
make.bat test             # Run tests
make.bat clean            # Clean up temporary files
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

## Troubleshooting

### Common Issues

#### Windows
- **"make is not recognized"**: Use `make.bat` instead or install make via winget
- **Python not found**: Add Python to PATH during installation
- **Permission errors**: Run Command Prompt as Administrator

#### macOS
- **"make: command not found"**: Install Xcode Command Line Tools: `xcode-select --install`
- **Python version issues**: Use pyenv or Homebrew Python

#### Linux
- **Package not found**: Update package lists: `sudo apt update` (Ubuntu) or `sudo yum update` (CentOS)
- **Permission denied**: Use `sudo` for system-wide installations

### Getting Help

1. Check the [uv documentation](https://docs.astral.sh/uv/)
2. Review [FastAPI documentation](https://fastapi.tiangolo.com/)
3. Check your Python version: `python --version`
4. Verify uv installation: `uv --version`

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

## Step-by-Step Guide: Setting Up and Using the Backend

### 1. Obtain Required API Keys

- **NewsAPI**: Register at [newsapi.org](https://newsapi.org/register) and get your API key.
- **Twitter API** (optional, for future social media support): Apply at [developer.twitter.com](https://developer.twitter.com/en/portal/dashboard) and create a project/app to get your Bearer Token.
- Place your API keys in a `.env` file (see `env.example`). Example:
  ```env
  NEWSAPI_KEY=your_newsapi_key_here
  TWITTER_BEARER_TOKEN=your_twitter_token_here
  ```

### 2. Start the Backend and Access Swagger UI

- Install dependencies (see earlier sections for `uv` or `pip` instructions).
- Start the server:
  ```bash
  python start_server.py
  # or
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- Open your browser and go to [http://localhost:8000/docs](http://localhost:8000/docs) to access the Swagger UI (interactive API documentation).

### 3. Gather News from All Sources

- (Optional) Add or activate sources using the `/sources` POST endpoint if you want to add more feeds/APIs.
- Trigger data gathering from all sources:
  ```bash
  curl -X POST "http://localhost:8000/gather"
  ```
- Or gather from a single source (replace `"BBC RSS"` with your source name):
  ```bash
  curl -X POST "http://localhost:8000/gather/single" -H "Content-Type: application/json" -d '"BBC RSS"'
  ```

### 4. Check if News is Stored

- List all articles:
  ```bash
  curl -X GET "http://localhost:8000/articles"
  ```
- Filter by source:
  ```bash
  curl -X GET "http://localhost:8000/articles?source=BBC%20RSS"
  ```
- Check the count:
  ```bash
  curl -X GET "http://localhost:8000/articles/count"
  ```

### 5. News Persistence

- All gathered news articles are now also saved in a file called `articles.json` in the project root.
- You can open this file to view all stored articles, or reload them if you restart the backend (future enhancement).
