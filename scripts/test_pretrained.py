#!/usr/bin/env python3
"""
Test pre-trained high-performance models for immediate use
This is the quickest path to meeting performance targets (≥90% accuracy, ≥95% F1)
"""

import sys
import os
from pathlib import Path
import time
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from app.dataset_manager import dataset_manager

def test_pretrained_model(model_name="Pulk17/Fake-News-Detection", test_size=100):
    """Test a pre-trained model on ISOT dataset"""
    print(f"\n" + "=" * 60)
    print(f"Testing Model: {model_name}")
    print("=" * 60)
    
    # Load the model
    print(f"\n⏳ Loading model...")
    start_time = time.time()
    
    try:
        # Try to load as pipeline first (simpler)
        classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load test data
    print(f"\n⏳ Loading ISOT test data...")
    df = dataset_manager.load_isot_dataset()
    
    if df is None:
        print("❌ ISOT dataset not found. Run: python scripts/download_isot.py")
        return None
    
    # Take a test sample
    test_df = df.sample(n=min(test_size, len(df)), random_state=42)
    print(f"✅ Loaded {len(test_df)} test samples")
    
    # Run predictions
    print(f"\n⏳ Running predictions...")
    predictions = []
    pred_labels = []
    true_labels = test_df['label'].tolist()
    
    start_time = time.time()
    for idx, row in test_df.iterrows():
        text = row['statement']
        result = classifier(text)
        
        # Map prediction to binary label
        pred_label = result[0]['label'].upper()
        if pred_label in ['FAKE', 'FALSE', '1']:
            pred_labels.append(1)
        elif pred_label in ['REAL', 'TRUE', '0']:
            pred_labels.append(0)
        else:
            # Default mapping based on score
            pred_labels.append(1 if result[0]['score'] > 0.5 else 0)
        
        predictions.append({
            'text': text[:100],
            'true_label': row['label'],
            'pred_label': pred_labels[-1],
            'confidence': result[0]['score']
        })
    
    pred_time = time.time() - start_time
    avg_time = pred_time / len(test_df) * 1000  # ms per prediction
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary'
    )
    
    # Display results
    print(f"\n" + "=" * 60)
    print("📊 Performance Metrics")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.2%} {'✅' if accuracy >= 0.90 else '❌'} (Target: ≥90%)")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%} {'✅' if f1 >= 0.95 else '❌'} (Target: ≥95%)")
    print(f"\nInference Speed: {avg_time:.1f}ms per text")
    print(f"Total Time: {pred_time:.2f} seconds for {len(test_df)} samples")
    
    # Show classification report
    print(f"\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(
        true_labels, pred_labels,
        target_names=['Real (0)', 'Fake (1)']
    ))
    
    # Show sample predictions
    print(f"\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    for i, pred in enumerate(predictions[:5]):
        symbol = "✅" if pred['true_label'] == pred['pred_label'] else "❌"
        print(f"\n{symbol} Sample {i+1}:")
        print(f"   Text: {pred['text']}...")
        print(f"   True: {pred['true_label']} | Predicted: {pred['pred_label']} | Confidence: {pred['confidence']:.3f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_ms': avg_time,
        'test_size': len(test_df)
    }

def compare_models():
    """Compare multiple pre-trained models"""
    models_to_test = [
        "Pulk17/Fake-News-Detection",  # Primary choice - 99.58% reported accuracy
        # "hamzab/roberta-fake-news-classification",  # Alternative - 100% training accuracy
        # "jy46604790/Fake-News-Bert-Detect",  # RoBERTa-base option
    ]
    
    print("\n" + "=" * 60)
    print("🔬 MODEL COMPARISON")
    print("=" * 60)
    print(f"Testing {len(models_to_test)} pre-trained models on ISOT dataset")
    print("Performance targets: ≥90% accuracy, ≥95% F1-score")
    
    results = []
    for model_name in models_to_test:
        try:
            result = test_pretrained_model(model_name, test_size=500)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n⚠️ Error testing {model_name}: {e}")
    
    # Summary comparison
    if results:
        print("\n" + "=" * 60)
        print("📊 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<40} {'Acc':<8} {'F1':<8} {'Speed':<10} {'Status'}")
        print("-" * 70)
        
        for r in results:
            meets_targets = r['accuracy'] >= 0.90 and r['f1'] >= 0.95
            status = "✅ PASS" if meets_targets else "❌ FAIL"
            model_short = r['model'].split('/')[-1][:38]
            print(f"{model_short:<40} {r['accuracy']:.2%}    {r['f1']:.2%}    {r['avg_inference_ms']:.1f}ms     {status}")
        
        # Find best model
        best_model = max(results, key=lambda x: (x['f1'], x['accuracy']))
        print(f"\n🏆 Best Model: {best_model['model']}")
        print(f"   F1-Score: {best_model['f1']:.2%}")
        print(f"   Accuracy: {best_model['accuracy']:.2%}")
        
        # Save results
        output_file = Path("processed_data/pretrained_model_results.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")

def quick_test():
    """Quick test with a single example"""
    print("\n" + "=" * 60)
    print("⚡ QUICK TEST")
    print("=" * 60)
    
    test_texts = [
        "Scientists discover breakthrough cure for cancer that doctors don't want you to know!",
        "The Federal Reserve announced a quarter-point interest rate increase today.",
        "You won't believe what this celebrity did! Number 7 will shock you!",
        "The president signed the infrastructure bill into law this morning.",
    ]
    
    print("Loading model...")
    classifier = pipeline(
        "text-classification",
        model="Pulk17/Fake-News-Detection",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print("\nTest predictions:")
    print("-" * 40)
    for text in test_texts:
        result = classifier(text)
        label = result[0]['label']
        confidence = result[0]['score']
        emoji = "🚫" if label.upper() in ['FAKE', 'FALSE'] else "✅"
        print(f"\n{emoji} {label} (confidence: {confidence:.3f})")
        print(f"   Text: {text[:80]}...")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pre-trained fake news detection models")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--model", default="Pulk17/Fake-News-Detection", help="Model to test")
    parser.add_argument("--samples", type=int, default=500, help="Number of test samples")
    
    args = parser.parse_args()
    
    print("\n🚀 Pre-trained Model Testing for Fake News Detection")
    print("=" * 60)
    print("Goal: Achieve ≥90% accuracy and ≥95% F1-score")
    print("Strategy: Use high-performance pre-trained models")
    print("=" * 60)
    
    if args.quick:
        quick_test()
    elif args.compare:
        compare_models()
    else:
        result = test_pretrained_model(args.model, args.samples)
        if result and result['accuracy'] >= 0.90 and result['f1'] >= 0.95:
            print("\n" + "🎉" * 20)
            print("🎉 SUCCESS! Performance targets met!")
            print(f"🎉 This model can be used immediately in production")
            print("🎉" * 20)
            print("\nNext steps:")
            print("1. Integrate this model into app/model_service_v2.py")
            print("2. Update API to use the high-performance model")
            print("3. Run full evaluation on entire test set")

if __name__ == "__main__":
    main()
