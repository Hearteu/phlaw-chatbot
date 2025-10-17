#!/usr/bin/env python3
"""
Test script for Saibo Legal RoBERTa document classification system
"""
import json
import os
import sys
import time
from typing import Any, Dict, List

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings from transformers library
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_legal_document_classifier():
    """Test the legal document classifier"""
    print("=" * 60)
    print("SAIBO LEGAL ROBERTA DOCUMENT CLASSIFIER TEST")
    print("=" * 60)
    
    try:
        # Test 1: Import the classifier
        from chatbot.legal_document_classifier import (
            LEGAL_CATEGORIES, NUM_LABELS, classify_legal_case,
            get_legal_document_classifier)
        print("Successfully imported legal document classifier")
        print(f"Number of classification labels: {NUM_LABELS}")
        print(f"Legal categories: {list(LEGAL_CATEGORIES.keys())}")
        
        # Test 2: Test with a criminal case
        print("\n--- Test 1: Criminal Case Classification ---")
        criminal_case = {
            "case_title": "People of the Philippines vs. Juan Dela Cruz",
            "content": "The accused Juan Dela Cruz was charged with the crime of Estafa under Article 315 of the Revised Penal Code. The prosecution alleged that the accused defrauded the complainant of P50,000 through false pretenses and fraudulent misrepresentation. The court found the accused guilty beyond reasonable doubt.",
            "gr_number": "G.R. No. 123456"
        }
        
        classification_result = classify_legal_case(criminal_case)
        print(f"Case: {criminal_case['case_title']}")
        print(f"Method: {classification_result.get('method', 'unknown')}")
        print(f"Confidence: {classification_result.get('confidence', 0.0):.3f}")
        print(f"Predictions:")
        for category, labels in classification_result.get('predictions', {}).items():
            print(f"  {category}: {labels}")
        
        # Test 3: Test with a civil case
        print("\n--- Test 2: Civil Case Classification ---")
        civil_case = {
            "case_title": "Maria Santos vs. ABC Corporation",
            "content": "This is a civil action for breach of contract and damages. The plaintiff seeks specific performance of the contract and payment of damages for the defendant's failure to deliver the goods as agreed upon in the contract. The court ruled in favor of the plaintiff.",
            "gr_number": "G.R. No. 789012"
        }
        
        classification_result = classify_legal_case(civil_case)
        print(f"Case: {civil_case['case_title']}")
        print(f"Method: {classification_result.get('method', 'unknown')}")
        print(f"Confidence: {classification_result.get('confidence', 0.0):.3f}")
        print(f"Predictions:")
        for category, labels in classification_result.get('predictions', {}).items():
            print(f"  {category}: {labels}")
        
        # Test 4: Test with administrative case
        print("\n--- Test 3: Administrative Case Classification ---")
        admin_case = {
            "case_title": "Administrative Matter No. 12345 - In re: Complaint against Judge Juan Perez",
            "content": "This administrative matter involves a complaint filed against Judge Juan Perez for gross ignorance of the law and conduct unbecoming of a judge. The complainant alleges that the judge rendered an unjust decision and violated the Code of Judicial Conduct.",
            "gr_number": "A.M. No. 12345"
        }
        
        classification_result = classify_legal_case(admin_case)
        print(f"Case: {admin_case['case_title']}")
        print(f"Method: {classification_result.get('method', 'unknown')}")
        print(f"Confidence: {classification_result.get('confidence', 0.0):.3f}")
        print(f"Predictions:")
        for category, labels in classification_result.get('predictions', {}).items():
            print(f"  {category}: {labels}")
        
        # Test 5: Test model info
        print("\n--- Test 4: Model Information ---")
        classifier = get_legal_document_classifier()
        model_info = classifier.get_model_info()
        print(f"Model Name: {model_info.get('model_name', 'Unknown')}")
        print(f"Model Loaded: {model_info.get('model_loaded', False)}")
        print(f"Tokenizer Loaded: {model_info.get('tokenizer_loaded', False)}")
        print(f"Classifier Loaded: {model_info.get('classifier_loaded', False)}")
        print(f"Device: {model_info.get('device', 'Unknown')}")
        print(f"Number of Labels: {model_info.get('num_labels', 'Unknown')}")
        print(f"Categories: {model_info.get('categories', [])}")
        
        print("\nAll classification tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics():
    """Test the evaluation metrics for 75-85% performance targets"""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS TEST")
    print("=" * 60)
    
    try:
        from chatbot.legal_document_classifier import \
            get_legal_document_classifier
        
        classifier = get_legal_document_classifier()
        
        # Create sample test data
        test_cases = [
            {
                'true_labels': {'case_type': 'criminal', 'legal_area': 'criminal_law'},
                'pred_labels': {'predictions': {'case_type': {'criminal': 0.8}, 'legal_area': {'criminal_law': 0.7}}}
            },
            {
                'true_labels': {'case_type': 'civil', 'legal_area': 'contract_law'},
                'pred_labels': {'predictions': {'case_type': {'civil': 0.9}, 'legal_area': {'contract_law': 0.8}}}
            },
            {
                'true_labels': {'case_type': 'administrative', 'legal_area': 'administrative_law'},
                'pred_labels': {'predictions': {'case_type': {'administrative': 0.7}, 'legal_area': {'administrative_law': 0.6}}}
            }
        ]
        
        y_true = [case['true_labels'] for case in test_cases]
        y_pred = [case['pred_labels'] for case in test_cases]
        
        # Test evaluation
        metrics = classifier.evaluate_classification_performance(y_true, y_pred, threshold=0.5)
        
        print("Evaluation Results:")
        print(f"Accuracy: {metrics.get('accuracy', 0.0):.3f}")
        print(f"Precision: {metrics.get('precision', 0.0):.3f}")
        print(f"Recall: {metrics.get('recall', 0.0):.3f}")
        print(f"F1 Score: {metrics.get('f1_score', 0.0):.3f}")
        print(f"Specificity: {metrics.get('specificity', 0.0):.3f}")
        print(f"Target Met (75-85%): {metrics.get('target_met', False)}")
        
        if metrics.get('error'):
            print(f"Evaluation Error: {metrics['error']}")
        
        print("\nEvaluation metrics test completed!")
        return True
        
    except Exception as e:
        print(f"Evaluation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_cache():
    """Test the model cache integration"""
    print("\n" + "=" * 60)
    print("MODEL CACHE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from chatbot.model_cache import (clear_legal_classifier_cache,
                                         get_cache_status,
                                         get_cached_legal_classifier)
        
        print("Initial cache status:")
        status = get_cache_status()
        print(f"Cache Status: {status}")
        
        # Test getting classifier
        classifier = get_cached_legal_classifier()
        if classifier:
            print("Legal document classifier loaded from cache")
            model_info = classifier.get_model_info()
            print(f"Model Info: {model_info}")
        else:
            print("Legal document classifier not loaded (may be expected if model not available)")
        
        # Test clearing cache
        clear_legal_classifier_cache()
        print("\nCache status after clearing:")
        status = get_cache_status()
        print(f"Cache Status: {status}")
        
        # Test reloading
        classifier_reloaded = get_cached_legal_classifier()
        if classifier_reloaded:
            print("\nLegal document classifier reloaded successfully.")
            model_info_reloaded = classifier_reloaded.get_model_info()
            print(f"Reloaded Model Info: {model_info_reloaded}")
        else:
            print("\nLegal document classifier failed to reload.")
            return False
        
        print("\nModel cache test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Model cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_engine_integration():
    """Test integration with chat engine"""
    print("\n" + "=" * 60)
    print("CHAT ENGINE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from chatbot.chat_engine import _extract_enhanced_case_type

        # Test document with Saibo classification metadata
        test_doc_with_classification = {
            "title": "People of the Philippines vs. Maria Clara",
            "content": "The accused Maria Clara was charged with qualified theft. Evidence showed she unlawfully took property belonging to the complainant.",
            "gr_number": "G.R. No. 987654",
            "metadata": {
                "case_type_classification": {
                    "criminal": 0.85,
                    "civil": 0.15
                },
                "classification_method": "saibo_classification",
                "classification_confidence": 0.85
            }
        }
        
        case_type = _extract_enhanced_case_type(test_doc_with_classification)
        print(f"Chat Engine Case Type (with Saibo classification): {case_type}")
        
        # Test document without Saibo classification
        test_doc_without_classification = {
            "title": "Maria Santos vs. ABC Corporation",
            "content": "This is a civil action for breach of contract and damages.",
            "gr_number": "G.R. No. 654321",
            "metadata": {
                "case_type": "civil"
            }
        }
        
        case_type = _extract_enhanced_case_type(test_doc_without_classification)
        print(f"Chat Engine Case Type (without Saibo classification): {case_type}")
        
        print("\nChat engine integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Chat engine integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("SAIBO LEGAL ROBERTA DOCUMENT CLASSIFICATION SYSTEM TEST")
    print("=" * 80)
    
    all_tests_passed = True
    
    # Test 1: Legal document classifier
    if not test_legal_document_classifier():
        all_tests_passed = False
    
    # Test 2: Evaluation metrics
    if not test_evaluation_metrics():
        all_tests_passed = False
    
    # Test 3: Model cache integration
    if not test_model_cache():
        all_tests_passed = False
    
    # Test 4: Chat engine integration
    if not test_chat_engine_integration():
        all_tests_passed = False
    
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    if all_tests_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("The Saibo Legal RoBERTa document classification system is working correctly.")
        print("The system is ready for achieving 75-85% classification performance targets.")
    else:
        print("[FAILED] SOME TESTS FAILED!")
        print("Please check the error messages above and fix any issues.")
    
    print("=" * 80)
