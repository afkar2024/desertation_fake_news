#!/usr/bin/env python3
"""
Debug script to understand the label mapping issue with pre-trained models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import pipeline
import torch

def test_model_outputs():
    """Test what the model actually outputs for known fake/real news"""
    
    # Test cases - clearly fake and clearly real news
    test_cases = [
        {
            "text": "Bill Gates will give you $5000 if you share this Facebook post!",
            "expected": "FAKE",
            "type": "Classic scam"
        },
        {
            "text": "Scientists discover miracle cure that doctors hate - click here now!",
            "expected": "FAKE",
            "type": "Clickbait"
        },
        {
            "text": "You won't believe what happened next! Number 7 will shock you!",
            "expected": "FAKE", 
            "type": "Clickbait"
        },
        {
            "text": "The Federal Reserve announced a quarter-point interest rate increase today following their regular policy meeting.",
            "expected": "REAL",
            "type": "Normal news"
        },
        {
            "text": "The president signed the infrastructure bill into law this morning in a ceremony at the White House.",
            "expected": "REAL",
            "type": "Normal news"
        }
    ]
    
    # Test with Pulk17 model
    print("=" * 60)
    print("Testing Pulk17/Fake-News-Detection Model")
    print("=" * 60)
    
    try:
        classifier = pipeline(
            "text-classification",
            model="Pulk17/Fake-News-Detection",
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("\nModel loaded successfully")
        print("-" * 40)
        
        for i, test in enumerate(test_cases, 1):
            result = classifier(test["text"])
            
            print(f"\nTest {i}: {test['type']}")
            print(f"Text: {test['text'][:80]}...")
            print(f"Expected: {test['expected']}")
            print(f"Raw output: {result}")
            print(f"Label: {result[0]['label']}")
            print(f"Score: {result[0]['score']:.3f}")
            
            # Check if correct
            actual_label = result[0]['label'].upper()
            if 'FAKE' in actual_label or 'FALSE' in actual_label:
                actual = "FAKE"
            elif 'REAL' in actual_label or 'TRUE' in actual_label:
                actual = "REAL"
            elif 'LABEL_1' in actual_label:
                actual = "FAKE"
            elif 'LABEL_0' in actual_label:
                actual = "REAL"
            else:
                actual = "UNKNOWN"
            
            symbol = "✅" if actual == test['expected'] else "❌"
            print(f"Result: {symbol} Interpreted as: {actual}")
            
    except Exception as e:
        print(f"Error loading Pulk17 model: {e}")
    
    # Test alternative interpretation
    print("\n" + "=" * 60)
    print("Testing if labels might be inverted")
    print("=" * 60)
    
    print("\nIf we assume the model outputs are INVERTED:")
    print("(i.e., model says REAL when it means FAKE)")
    print("-" * 40)
    
    # This would explain why obvious fake news is being labeled as REAL

def test_alternative_models():
    """Test other models to see if they work better"""
    
    alternative_models = [
        "roberta-base-openai-detector",  # OpenAI GPT-2 detector
        "dslim/bert-base-NER",  # Just for testing
    ]
    
    print("\n" + "=" * 60)
    print("Testing Alternative Models")
    print("=" * 60)
    
    test_text = "Bill Gates will give you money if you share this post!"
    
    for model_name in alternative_models:
        print(f"\nTesting: {model_name}")
        print("-" * 40)
        try:
            classifier = pipeline("text-classification", model=model_name)
            result = classifier(test_text)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Could not test {model_name}: {e}")

def main():
    print("\n🔍 Debugging Pre-trained Model Labels")
    print("=" * 60)
    print("This script tests what labels the models actually output")
    print("to understand why fake news is being classified as real.")
    print("=" * 60)
    
    test_model_outputs()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("\nBased on the results above, we need to:")
    print("1. Check if the model's labels are inverted")
    print("2. Try a different pre-trained model")
    print("3. Or fix the label mapping in our code")
    
    print("\nPossible solutions:")
    print("- Invert the label mapping (REAL->1, FAKE->0)")
    print("- Use a different model that works correctly")
    print("- Check the model card on HuggingFace for correct usage")

if __name__ == "__main__":
    main()
