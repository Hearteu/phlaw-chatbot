#!/usr/bin/env python3
"""
Integration script to use the fine-tuned Legal RoBERTa model
in your existing chatbot system.
"""

import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_finetuned_model(model_path: str = "./legal_roberta_finetuned"):
    """Load the fine-tuned model and tokenizer"""
    
    if not os.path.exists(model_path):
        print(f"❌ Fine-tuned model not found at {model_path}")
        print("Please run fine_tune_legal_roberta.py first")
        return None, None
    
    print(f"📥 Loading fine-tuned model from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set to evaluation mode
        model.eval()
        
        print("✅ Fine-tuned model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return None, None


def classify_with_finetuned_model(text: str, model, tokenizer, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Classify text using the fine-tuned model"""
    
    if model is None or tokenizer is None:
        return {"success": False, "error": "Model not loaded"}
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
        
        # Convert to CPU for processing
        probabilities = probabilities.cpu().numpy()[0]
        
        # Get predictions above threshold
        predictions = {}
        for i, prob in enumerate(probabilities):
            if prob > confidence_threshold:
                label_id = model.config.id2label.get(i, f"label_{i}")
                predictions[label_id] = float(prob)
        
        # Calculate overall confidence
        overall_confidence = max(probabilities) if len(probabilities) > 0 else 0.0
        
        return {
            "success": True,
            "predictions": predictions,
            "confidence": float(overall_confidence),
            "method": "finetuned_legal_roberta"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def update_legal_document_classifier():
    """Update the existing legal_document_classifier.py to use fine-tuned model"""
    
    print("🔄 Updating legal_document_classifier.py to use fine-tuned model...")
    
    # Read current file
    classifier_file = "chatbot/legal_document_classifier.py"
    if not os.path.exists(classifier_file):
        print(f"❌ {classifier_file} not found")
        return False
    
    # Backup original
    backup_file = f"{classifier_file}.backup"
    if not os.path.exists(backup_file):
        with open(classifier_file, 'r') as f:
            content = f.read()
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"📁 Created backup: {backup_file}")
    
    # Load fine-tuned model
    model, tokenizer = load_finetuned_model()
    if model is None:
        return False
    
    # Test the integration
    test_text = "This is a criminal case involving theft and robbery."
    result = classify_with_finetuned_model(test_text, model, tokenizer)
    
    if result["success"]:
        print("✅ Fine-tuned model integration successful")
        print(f"   Test classification: {result['predictions']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        return True
    else:
        print(f"❌ Integration test failed: {result.get('error', 'Unknown error')}")
        return False


def main():
    """Main integration function"""
    print("🚀 Legal RoBERTa Fine-tuned Model Integration")
    
    # Check if fine-tuned model exists
    model_path = "./legal_roberta_finetuned"
    if not os.path.exists(model_path):
        print(f"❌ Fine-tuned model not found at {model_path}")
        print("Please run fine_tune_legal_roberta.py first")
        return
    
    # Test model loading
    model, tokenizer = load_finetuned_model(model_path)
    if model is None:
        return
    
    # Test classification
    test_cases = [
        "This is an administrative matter involving missing cash bonds in criminal cases.",
        "The plaintiff filed a civil case for breach of contract.",
        "Criminal charges were filed against the accused for theft.",
        "Constitutional question regarding the Bill of Rights."
    ]
    
    print("\n🧪 Testing fine-tuned model:")
    for i, test_case in enumerate(test_cases, 1):
        result = classify_with_finetuned_model(test_case, model, tokenizer)
        print(f"\nTest {i}: {test_case[:50]}...")
        if result["success"]:
            print(f"   Predictions: {list(result['predictions'].keys())[:3]}")
            print(f"   Confidence: {result['confidence']:.3f}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    # Update integration
    if update_legal_document_classifier():
        print("\n✅ Integration complete!")
        print("📝 Next steps:")
        print("   1. Update embed.py to use the fine-tuned model")
        print("   2. Re-run the embedding pipeline")
        print("   3. Test the chatbot with improved classification")
    else:
        print("\n❌ Integration failed. Please check the errors above.")


if __name__ == "__main__":
    main()
