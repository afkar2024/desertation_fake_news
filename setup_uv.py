#!/usr/bin/env python3
"""
Setup script for the Adaptive Fake News Detector using uv
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and handle errors"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_uv_installed():
    """Check if uv is installed"""
    return run_command("uv --version", check=False)

def main():
    """Main setup function"""
    print("🚀 Setting up Adaptive Fake News Detector with uv")
    print("=" * 60)
    
    # Check if uv is installed
    if not check_uv_installed():
        print("❌ uv is not installed!")
        print("📦 Install uv from: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    print("✅ uv is installed")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    if not run_command("uv pip install -e ."):
        print("❌ Failed to install dependencies")
        return False
    print("✅ Dependencies installed")
    
    print("\n🎉 Setup complete!")
    print("Run: python start_server.py")
    
    return True

if __name__ == "__main__":
    main() 