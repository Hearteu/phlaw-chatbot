#!/usr/bin/env python3
"""
Test script to verify the fine-tuned Legal RoBERTa integration in embed.py
"""

import os
import sys

# Add the chatbot directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'chatbot'))

def test_finetuned_model_loading():
    """Test if the fine-tuned model loads correctly"""
    try:
        from embed import classify_with_finetuned_model, load_finetuned_model
        
        print("🧪 Testing fine-tuned model loading...")
        model, tokenizer = load_finetuned_model()
        
        if model is None or tokenizer is None:
            print("❌ Failed to load fine-tuned model")
            return False
        
        print("✅ Fine-tuned model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return False

def test_classification():
    """Test classification with sample case data"""
    try:
        from embed import classify_with_finetuned_model
        
        print("\n🧪 Testing classification...")
        
        # Sample case data
        test_case = {
            'gr_number': 'G.R. No. 123456',
            'case_title': 'People of the Philippines vs. John Doe',
            'clean_text': 'This is a criminal case involving theft and robbery. The accused was charged with violating Article 308 of the Revised Penal Code.',
            'case_type': 'criminal',
            'division': 'First Division'
        }
        
        result = classify_with_finetuned_model(test_case)
        
        if result.get('success', False):
            print("✅ Classification successful")
            print(f"   Method: {result.get('method', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
            print(f"   Predictions: {list(result.get('predictions', {}).keys())[:5]}")
            return True
        else:
            print(f"❌ Classification failed: {result.get('error', 'unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during classification test: {e}")
        return False

def test_model_info():
    """Test model configuration and info"""
    try:
        from embed import load_finetuned_model
        
        print("\n🧪 Testing model info...")
        model, tokenizer = load_finetuned_model()
        
        if model is None:
            print("❌ Model not loaded")
            return False
        
        print(f"✅ Model loaded:")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Number of labels: {model.config.num_labels}")
        print(f"   Model name: {model.config._name_or_path}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Fine-tuned Legal RoBERTa Integration")
    print("=" * 50)
    
    tests = [
        test_finetuned_model_loading,
        test_model_info,
        test_classification
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fine-tuned model integration is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Run: python embed.py")
        print("   2. Monitor classification confidence scores")
        print("   3. Verify improved case type detection")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
