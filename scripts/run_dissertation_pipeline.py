#!/usr/bin/env python3
"""
End-to-end pipeline for dissertation demo:
  1) Ensure LIAR dataset is available
  2) Fine-tune transformer on LIAR (calls scripts/train_liar.py)
  3) Reload backend model with the new MODEL_PATH
  4) Evaluate on a subset and save metrics report

Usage:
  python scripts/run_dissertation_pipeline.py --api http://localhost:8000 --model-dir models/liar_finetuned --epochs 2 --limit 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import sleep
from typing import Any, Dict

import requests


def ensure_dataset_liar() -> None:
    # We can trigger download via API or directly run the CLI script
    # Prefer API: /datasets/download/liar (background). We'll call CLI to be synchronous instead.
    try:
        from app.dataset_manager import dataset_manager  # type: ignore

        info = dataset_manager.get_dataset_info("liar")
        if info and Path(info.get("local_path", "")).exists():
            print("LIAR dataset already available at:", info["local_path"]) 
            return
        print("Downloading LIAR dataset...")
        ok = dataset_manager.download_liar_dataset()
        if not ok:
            print("Failed to download LIAR dataset", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print("Dataset check/download failed:", e, file=sys.stderr)
        sys.exit(1)


def run_training(model_dir: str, epochs: int, model_name: str) -> None:
    cmd = [
        sys.executable,
        "scripts/train_liar.py",
        "--epochs",
        str(epochs),
        "--output-dir",
        model_dir,
        "--model-name",
        model_name,
    ]
    print("Running training:", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("Training failed", file=sys.stderr)
        sys.exit(proc.returncode)


def api_reload_model(api_base: str, model_dir: str) -> None:
    url = f"{api_base}/model/reload"
    resp = requests.post(url, json={"model_source": model_dir}, timeout=120)
    resp.raise_for_status()
    print("Reload response:", resp.json())


def api_evaluate(api_base: str, limit: int) -> Dict[str, Any]:
    url = f"{api_base}/datasets/evaluate/liar"
    body = {
        "text_column": "statement",
        "label_column": "label",
        "limit": limit,
        "compare_baseline": True,
    }
    resp = requests.post(url, json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Run dissertation demo pipeline")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--model-dir", default="models/liar_finetuned", help="Local output dir for model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    ensure_dataset_liar()
    run_training(args.model_dir, args.epochs, args.model_name)

    # Give the backend a moment to be ready if just started
    try:
        api_reload_model(args.api, args.model_dir)
    except requests.RequestException as e:
        print("Model reload failed:", e, file=sys.stderr)
        sys.exit(1)

    try:
        result = api_evaluate(args.api, args.limit)
        print("\nEvaluation Result:")
        for k, v in result.items():
            print(f"- {k}: {v}")
    except requests.RequestException as e:
        print("Evaluation failed:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


