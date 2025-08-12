#!/usr/bin/env python3
"""
Minimal LIAR fine-tuning script for dissertation demo.

Trains a binary classifier (real=0, fake=1) on LIAR using Hugging Face Trainer,
saves the model locally, and writes a metrics report under processed_data/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Local imports
from app.dataset_manager import dataset_manager
from app.reports import write_metrics_report


def map_liar_to_binary(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    mapping = {
        "true": 0,
        "mostly-true": 0,
        "half-true": 0,
        "barely-true": 1,
        "false": 1,
        "pants-fire": 1,
        "pants-on-fire": 1,
    }
    df = df.copy()
    df["binary_label"] = df[label_col].map(mapping)
    df = df.dropna(subset=["binary_label", "statement"]).reset_index(drop=True)
    df["binary_label"] = df["binary_label"].astype(int)
    return df[["statement", "binary_label"]]


@dataclass
class TextDataset(torch.utils.data.Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on LIAR dataset")
    parser.add_argument("--model-name", default="distilbert-base-uncased", type=str)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--output-dir", default="models/liar_finetuned", type=str)
    parser.add_argument("--max-length", default=256, type=int)
    args = parser.parse_args()

    # Load LIAR
    df = dataset_manager.load_liar_dataset()
    if df is None or df.empty:
        raise SystemExit("LIAR dataset not found. Download it first via download_and_preprocess.py --dataset liar")

    df = map_liar_to_binary(df)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["binary_label"]
    )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(
        train_df["statement"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        test_df["statement"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    train_dataset = TextDataset(train_enc, train_df["binary_label"].values)
    eval_dataset = TextDataset(test_enc, test_df["binary_label"].values)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # Training args
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save model and tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Write metrics report
    report_path = Path("processed_data")
    write_metrics_report(
        {
            "dataset": "LIAR",
            "model": args.model_name,
            "accuracy": round(float(metrics.get("eval_accuracy", 0.0)), 4),
            "precision": round(float(metrics.get("eval_precision", 0.0)), 4),
            "recall": round(float(metrics.get("eval_recall", 0.0)), 4),
            "f1": round(float(metrics.get("eval_f1", 0.0)), 4),
            "saved_model_path": str(output_dir.resolve()),
        },
        report_path,
    )

    print("\n✅ Training complete.")
    print(f"📁 Model saved to: {output_dir.resolve()}")
    print("ℹ️  To use this model in the API, set in .env: MODEL_PATH=", output_dir.resolve())


if __name__ == "__main__":
    main()


