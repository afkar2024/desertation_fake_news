#!/usr/bin/env python3
"""
Startup script for the Adaptive Fake News Detector API
"""
import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Start the FastAPI server"""
    
    # Add the app directory to Python path
    app_dir = Path(__file__).parent / "app"
    sys.path.insert(0, str(app_dir))
    
    # Set default environment variables if not set
    os.environ.setdefault("HOST", "0.0.0.0")
    os.environ.setdefault("PORT", "8000")
    os.environ.setdefault("LOG_LEVEL", "info")
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info")
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print("=" * 60)
    print("🚀 Starting Adaptive Fake News Detector API")
    print("=" * 60)
    print(f"📡 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔧 Interactive API: http://{host}:{port}/redoc")
    print("=" * 60)
    print()
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Found .env file - loading environment variables")
    else:
        print("⚠️  No .env file found - using default settings")
        print("💡 Create a .env file to configure API keys and settings")
    
    print()
    print("🔑 To add API keys, create a .env file with:")
    print("   NEWSAPI_KEY=your_newsapi_key_here")
    print("   TWITTER_BEARER_TOKEN=your_twitter_token_here")
    print()
    
    try:
        # Start the server
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
            reload_dirs=["app"] if reload else None
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped gracefully")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 