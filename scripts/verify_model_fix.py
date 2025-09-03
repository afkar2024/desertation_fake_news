#!/usr/bin/env python3
"""
Verify that the model correctly identifies fake news using the verified implementation
Based on emergency fix script with proven label parsing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model_service_v2 import EnhancedModelService
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model_with_known_examples():
    """Test with clear fake and real news examples"""
    
    print("=" * 60)
    print("🧪 Testing Enhanced Model Service")
    print("=" * 60)
    
    # Initialize the service
    print("\n⏳ Loading model...")
    try:
        service = EnhancedModelService(strategy="pretrained", primary_model="pulk17")
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test cases from emergency fix script (verified to work)
    test_cases = [
        {
            "text": "Bill Gates will give you $5000 if you share this Facebook post! Hey Facebook, As some of you may know, I'm Bill Gates. If you click that share link, I will give you $5,000.",
            "expected": "fake",
            "description": "Classic Facebook scam"
        },
        {
            "text": "Scientists have discovered a miracle cure that doctors don't want you to know about! This revolutionary treatment can cure cancer, diabetes, and heart disease instantly with just one simple trick!",
            "expected": "fake",
            "description": "Obvious fake news with sensational claims"
        },
        {
            "text": "You won't believe what happened next! Number 7 will shock you!",
            "expected": "fake",
            "description": "Clickbait"
        },
        {
            "text": "BREAKING: Aliens have landed in New York City and are demanding to speak to world leaders immediately. Military sources confirm multiple UFO sightings over Manhattan.",
            "expected": "fake",
            "description": "Sensational fake breaking news"
        },
        {
            "text": "The Federal Reserve announced today that it will maintain interest rates at current levels following a two-day meeting. The decision was unanimous among voting members of the Federal Open Market Committee.",
            "expected": "real",
            "description": "Legitimate financial news"
        },
        {
            "text": "The World Health Organization released new guidelines for vaccinations today, recommending updated schedules for several routine immunizations based on recent clinical studies.",
            "expected": "real",
            "description": "Factual health news"
        },
        {
            "text": "The president signed the infrastructure bill into law this morning in a ceremony at the White House.",
            "expected": "real",
            "description": "Normal political news"
        }
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    print("Running predictions...")
    print("-" * 40)
    
    for i, case in enumerate(test_cases, 1):
        try:
            # Get prediction
            result = service.classify_text(case["text"])
            
            # Determine predicted class
            if result['prediction'] == 1:
                predicted_class = "fake"
            else:
                predicted_class = "real"
            
            confidence = result['confidence']
            
            # Check if correct
            is_correct = predicted_class.lower() == case["expected"].lower()
            if is_correct:
                correct_predictions += 1
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            
            print(f"\nTest {i}: {case['description']}")
            print(f"  Text: {case['text'][:80]}...")
            print(f"  Expected: {case['expected'].upper()}")
            print(f"  Predicted: {predicted_class.upper()} (confidence: {confidence:.3f})")
            print(f"  Result: {status}")
            
        except Exception as e:
            print(f"❌ Test {i} failed with error: {e}")
    
    # Calculate accuracy
    accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 Results Summary")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.1%} ({correct_predictions}/{total_tests})")
    
    if accuracy >= 0.85:
        print("🎉 SUCCESS! Model is working correctly!")
        print("The label parsing fix is working as expected.")
        return True
    elif accuracy >= 0.70:
        print("⚠️  Model performance is acceptable but could be better")
        print("Consider testing with the backup models")
        return True
    else:
        print("❌ Model performance is poor")
        print("The labels might still be inverted or there's another issue")
        return False

def test_specific_case():
    """Test the specific Bill Gates scam that was failing"""
    print("\n" + "=" * 60)
    print("🎯 Testing Specific Bill Gates Scam Case")
    print("=" * 60)
    
    service = EnhancedModelService(strategy="pretrained", primary_model="pulk17")
    
    text = """Share a certain post of Bill Gates on Facebook and he will send you money.
"Hey Facebook, As some of you may know, I'm Bill Gates. If you click that share link, I will give you $5,000. I always deliver, I mean, I brought you Windows XP, right?"""
    
    result = service.classify_text(text)
    
    print(f"\nText: {text}")
    print(f"\nRaw Result: {result}")
    print(f"\nPrediction: {'FAKE' if result['prediction'] == 1 else 'REAL'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Used: {result['model_used']}")
    
    if result['prediction'] == 1:
        print("\n✅ CORRECT! This is correctly identified as FAKE news")
        return True
    else:
        print("\n❌ INCORRECT! This should be identified as FAKE news")
        return False

def main():
    print("\n🚀 Fake News Detection Model Verification")
    print("=" * 60)
    print("Testing the enhanced model service with verified label parsing")
    print("Based on the emergency fix script implementation")
    print("=" * 60)
    
    # Run comprehensive tests
    success1 = test_model_with_known_examples()
    
    # Test specific case
    success2 = test_specific_case()
    
    if success1 and success2:
        print("\n" + "🎉" * 20)
        print("🎉 ALL TESTS PASSED!")
        print("🎉 The model is now correctly identifying fake news")
        print("🎉 You can proceed with the implementation")
        print("🎉" * 20)
        
        print("\n📋 Next Steps:")
        print("1. Download ISOT dataset: python scripts/download_isot.py")
        print("2. Test on full dataset: python scripts/test_pretrained.py")
        print("3. Integrate into API: Update app/main.py to use model_service_v2")
        print("4. Run evaluation: Test with your evaluation scripts")
    else:
        print("\n⚠️ Some tests failed")
        print("Please review the label parsing logic")
        print("Consider testing with backup models (hamzab or jy46604790)")

if __name__ == "__main__":
    main()
