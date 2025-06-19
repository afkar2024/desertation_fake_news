@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Available commands:
    echo   install     - Install basic dependencies
    echo   install-dev - Install with development dependencies
    echo   install-ml  - Install with ML dependencies
    echo   run         - Start the FastAPI server
    echo   test        - Run tests (when available)
    echo   format      - Format code with black
    echo   lint        - Run linting with ruff
    echo   clean       - Clean up temporary files
    goto :eof
)

if "%1"=="install" (
    echo Installing basic dependencies...
    uv pip install -e .
    goto :eof
)

if "%1"=="install-dev" (
    echo Installing with development dependencies...
    uv pip install -e ".[dev]"
    goto :eof
)

if "%1"=="install-ml" (
    echo Installing with ML dependencies...
    uv pip install -e ".[ml]"
    goto :eof
)

if "%1"=="run" (
    echo Starting the FastAPI server...
    python start_server.py
    goto :eof
)

if "%1"=="test" (
    echo Running tests...
    pytest tests/
    goto :eof
)

if "%1"=="format" (
    echo Formatting code...
    black .
    ruff check --fix .
    goto :eof
)

if "%1"=="lint" (
    echo Running linting...
    ruff check .
    black --check .
    goto :eof
)

if "%1"=="clean" (
    echo Cleaning up temporary files...
    for /r . %%f in (*.pyc) do del "%%f" 2>nul
    for /d /r . %%d in (__pycache__) do rmdir /s /q "%%d" 2>nul
    for /d /r . %%d in (*.egg-info) do rmdir /s /q "%%d" 2>nul
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    goto :eof
)

echo Unknown command: %1
echo Use 'make.bat' without arguments to see available commands. 