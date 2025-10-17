# legal_document_classifier.py — Legal document classification using Saibo Legal RoBERTa
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from transformers import (AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, AutoTokenizer)

# Legal categories for jurisprudence classification
LEGAL_CATEGORIES = {
    'case_type': [
        'civil', 'criminal', 'administrative', 'constitutional', 'labor',
        'commercial', 'family', 'property', 'tort', 'contract'
    ],
    'legal_area': [
        'contract_law', 'property_law', 'family_law', 'labor_law', 
        'criminal_law', 'administrative_law', 'constitutional_law',
        'commercial_law', 'tort_law', 'corporate_law', 'tax_law',
        'environmental_law', 'intellectual_property', 'employment_law'
    ],
    'document_section': [
        'facts', 'issues', 'ruling', 'ratio_decidendi', 'disposition',
        'concurring_opinion', 'dissenting_opinion', 'syllabus',
        'case_summary', 'legal_precedent'
    ],
    'complexity_level': [
        'simple', 'moderate', 'complex', 'highly_complex'
    ],
    'jurisdiction_level': [
        'supreme_court', 'appellate_court', 'regional_trial_court',
        'municipal_trial_court', 'specialized_court'
    ]
}

# Create flattened label mapping for multi-label classification
ALL_LABELS = []
LABEL_TO_ID = {}
ID_TO_LABEL = {}

for category, labels in LEGAL_CATEGORIES.items():
    for label in labels:
        full_label = f"{category}:{label}"
        ALL_LABELS.append(full_label)
        LABEL_TO_ID[full_label] = len(ALL_LABELS) - 1
        ID_TO_LABEL[len(ALL_LABELS) - 1] = full_label

NUM_LABELS = len(ALL_LABELS)


class LegalDocumentClassifier:
    """Legal document classifier using Saibo Legal RoBERTa model"""
    
    def __init__(self, model_name: str = "Saibo-creator/legal-roberta-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.classifier_head = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """Load the Saibo Legal RoBERTa model and tokenizer"""
        try:
            print(f"Loading Saibo Legal RoBERTa model for classification: {self.model_name}")
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=os.path.join(os.path.dirname(__file__), "..", "model_cache", "hub")
            )
            
            # Load base model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=os.path.join(os.path.dirname(__file__), "..", "model_cache", "hub")
            )
            
            # Create classification head for multi-label classification
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.classifier_head = nn.Linear(
                self.config.hidden_size, 
                NUM_LABELS
            )
            
            # Move to device
            self.model.to(self.device)
            self.classifier_head.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            print(f"Saibo Legal RoBERTa classifier loaded in {load_time:.2f}s")
            print(f"Number of classification labels: {NUM_LABELS}")
            
        except Exception as e:
            print(f"Failed to load Saibo classifier: {e}")
            print("Will use rule-based classification as fallback")
            self.model = None
            self.tokenizer = None
            self.classifier_head = None
    
    def classify_legal_document(
        self, 
        text: str, 
        case_title: str = "", 
        gr_number: str = "",
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Classify legal document using Saibo model
        
        Args:
            text: Legal document text
            case_title: Case title (optional)
            gr_number: G.R. number (optional)
            max_length: Maximum sequence length
            
        Returns:
            Classification results with confidence scores
        """
        if not self.model or not self.tokenizer:
            return self._fallback_classification(text, case_title, gr_number)
        
        try:
            # Combine title and content for better classification
            combined_text = f"{case_title}\n{text}" if case_title else text
            
            # Tokenize input
            inputs = self.tokenizer(
                combined_text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use pooled output or mean of last hidden state
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    pooled_output = outputs.pooler_output
                else:
                    pooled_output = outputs.last_hidden_state.mean(dim=1)
                
                # Get classification logits
                logits = self.classifier_head(pooled_output)
                
                # Apply sigmoid for multi-label classification
                probabilities = torch.sigmoid(logits)
                
                # Get top predictions
                top_indices = torch.topk(probabilities, k=min(10, NUM_LABELS), dim=1)
                
                predictions = {}
                for i, (idx, prob) in enumerate(zip(top_indices.indices[0], top_indices.values[0])):
                    label = ID_TO_LABEL[idx.item()]
                    predictions[label] = prob.item()
                
                # Organize predictions by category
                organized_predictions = self._organize_predictions(predictions)
                
                return {
                    'success': True,
                    'method': 'saibo_classification',
                    'predictions': organized_predictions,
                    'raw_scores': predictions,
                    'confidence': max(predictions.values()) if predictions else 0.0,
                    'model_info': {
                        'model_name': self.model_name,
                        'num_labels': NUM_LABELS,
                        'device': str(self.device)
                    }
                }
                
        except Exception as e:
            print(f"Saibo classification failed: {e}")
            return self._fallback_classification(text, case_title, gr_number)
    
    def _organize_predictions(self, predictions: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Organize predictions by category"""
        organized = {category: {} for category in LEGAL_CATEGORIES.keys()}
        
        for label, score in predictions.items():
            if ':' in label:
                category, sub_label = label.split(':', 1)
                if category in organized:
                    organized[category][sub_label] = score
        
        # Sort by confidence for each category
        for category in organized:
            organized[category] = dict(
                sorted(organized[category].items(), key=lambda x: x[1], reverse=True)
            )
        
        return organized
    
    def _fallback_classification(
        self, 
        text: str, 
        case_title: str = "", 
        gr_number: str = ""
    ) -> Dict[str, Any]:
        """Fallback rule-based classification"""
        text_lower = text.lower()
        title_lower = case_title.lower()
        
        # Rule-based classification logic
        predictions = {
            'case_type': {},
            'legal_area': {},
            'document_section': {},
            'complexity_level': {},
            'jurisdiction_level': {}
        }
        
        # Case type classification
        criminal_keywords = ['criminal', 'penal', 'theft', 'robbery', 'murder', 'homicide', 'assault', 'fraud', 'estafa', 'violation', 'offense', 'crime', 'penalty', 'imprisonment']
        civil_keywords = ['civil', 'contract', 'agreement', 'obligation', 'breach', 'performance', 'consideration', 'parties', 'terms', 'conditions', 'stipulation', 'covenant']
        administrative_keywords = ['administrative', 'government', 'public', 'official', 'discipline', 'misconduct', 'duty', 'authority', 'jurisdiction', 'agency']
        
        # Score each case type
        criminal_score = sum(1 for keyword in criminal_keywords if keyword in text_lower or keyword in title_lower)
        civil_score = sum(1 for keyword in civil_keywords if keyword in text_lower or keyword in title_lower)
        admin_score = sum(1 for keyword in administrative_keywords if keyword in text_lower or keyword in title_lower)
        
        if criminal_score > 0:
            predictions['case_type']['criminal'] = min(criminal_score / 5.0, 1.0)
        if civil_score > 0:
            predictions['case_type']['civil'] = min(civil_score / 5.0, 1.0)
        if admin_score > 0:
            predictions['case_type']['administrative'] = min(admin_score / 5.0, 1.0)
        
        # Default to civil if no clear classification
        if not predictions['case_type']:
            predictions['case_type']['civil'] = 0.5
        
        # Jurisdiction level based on GR number
        if 'g.r. no.' in gr_number.lower() or 'gr no' in gr_number.lower():
            predictions['jurisdiction_level']['supreme_court'] = 0.9
        elif 'a.m.' in gr_number.lower():
            predictions['jurisdiction_level']['supreme_court'] = 0.8
        else:
            predictions['jurisdiction_level']['regional_trial_court'] = 0.6
        
        # Complexity level based on text length
        if len(text) > 10000:
            predictions['complexity_level']['highly_complex'] = 0.8
        elif len(text) > 5000:
            predictions['complexity_level']['complex'] = 0.7
        elif len(text) > 2000:
            predictions['complexity_level']['moderate'] = 0.6
        else:
            predictions['complexity_level']['simple'] = 0.7
        
        return {
            'success': True,
            'method': 'rule_based_fallback',
            'predictions': predictions,
            'raw_scores': {},
            'confidence': max([max(cat.values()) for cat in predictions.values() if cat], default=0.0),
            'model_info': {
                'model_name': 'rule_based_fallback',
                'num_labels': NUM_LABELS,
                'device': 'cpu'
            }
        }
    
    def evaluate_classification_performance(
        self, 
        y_true: List[Dict], 
        y_pred: List[Dict],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate classification performance against 75-85% targets
        
        Args:
            y_true: True labels (list of dictionaries with category:label mappings)
            y_pred: Predicted labels (list of dictionaries with category:label mappings)
            threshold: Confidence threshold for predictions
            
        Returns:
            Performance metrics
        """
        try:
            # Flatten predictions for evaluation
            y_true_flat = []
            y_pred_flat = []
            
            for true_labels, pred_labels in zip(y_true, y_pred):
                # Convert to binary vectors
                true_vector = [0.0] * NUM_LABELS
                pred_vector = [0.0] * NUM_LABELS
                
                # Fill true labels
                for category, label in true_labels.items():
                    full_label = f"{category}:{label}"
                    if full_label in LABEL_TO_ID:
                        true_vector[LABEL_TO_ID[full_label]] = 1.0
                
                # Fill predicted labels
                for category, labels in pred_labels.get('predictions', {}).items():
                    for label, score in labels.items():
                        if score >= threshold:
                            full_label = f"{category}:{label}"
                            if full_label in LABEL_TO_ID:
                                pred_vector[LABEL_TO_ID[full_label]] = 1.0
                
                y_true_flat.append(true_vector)
                y_pred_flat.append(pred_vector)
            
            # Convert to numpy arrays
            y_true_flat = np.array(y_true_flat)
            y_pred_flat = np.array(y_pred_flat)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true_flat, y_pred_flat)
            precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
            recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
            f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
            
            # Calculate specificity (True Negative Rate)
            tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
            fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Check if targets are met (75-85% range)
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity
            }
            
            target_met = all(0.75 <= score <= 0.85 for score in metrics.values())
            metrics['target_met'] = target_met
            
            return metrics
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
                'target_met': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'classifier_loaded': self.classifier_head is not None,
            'device': str(self.device),
            'num_labels': NUM_LABELS,
            'categories': list(LEGAL_CATEGORIES.keys()),
            'model_type': 'legal_document_classifier'
        }


# Global classifier instance
_classifier = None

def get_legal_document_classifier() -> LegalDocumentClassifier:
    """Get or create the global legal document classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = LegalDocumentClassifier()
    return _classifier

def classify_legal_case(
    case_data: Dict[str, Any], 
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to classify legal case data
    
    Args:
        case_data: Case data dictionary
        include_metadata: Whether to include metadata in results
        
    Returns:
        Classification results
    """
    classifier = get_legal_document_classifier()
    
    # Extract text components
    case_title = case_data.get("case_title", "") or case_data.get("title", "")
    case_content = case_data.get("content", "") or case_data.get("clean_text", "") or case_data.get("body", "")
    gr_number = case_data.get("gr_number", "") or case_data.get("metadata", {}).get("gr_number", "")
    
    # Classify the document
    classification_result = classifier.classify_legal_document(
        text=case_content,
        case_title=case_title,
        gr_number=gr_number
    )
    
    # Add metadata if requested
    if include_metadata:
        classification_result['input_metadata'] = {
            'case_title': case_title,
            'gr_number': gr_number,
            'content_length': len(case_content),
            'classification_timestamp': time.time()
        }
    
    return classification_result
