#!/usr/bin/env python3
"""
Minimal demo script to call the backend API for prediction and SHAP explanations.

Usage:
  python scripts/demo_api.py --text "Your news text here"
"""

import argparse
import json
import sys
from typing import Any, Dict

import requests


def predict(base_url: str, text: str) -> Dict[str, Any]:
    url = f"{base_url}/predict"
    resp = requests.post(url, json={"text": text})
    resp.raise_for_status()
    return resp.json()


def explain(base_url: str, text: str) -> Dict[str, Any]:
    url = f"{base_url}/explain/shap"
    resp = requests.post(url, json={"text": text})
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Demo API client")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--text", required=True, help="Input text to classify")
    args = parser.parse_args()

    try:
        pred = predict(args.base_url, args.text)
        print("Prediction:")
        print(json.dumps(pred, indent=2))

        exp = explain(args.base_url, args.text)
        print("\nSHAP Explanation (first 30 tokens):")
        preview = {
            "tokens": exp.get("tokens", [])[:30],
            "shap_values": exp.get("shap_values", [])[:30],
            "base_value": exp.get("base_value"),
        }
        print(json.dumps(preview, indent=2))
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        print(getattr(e.response, "text", ""), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


