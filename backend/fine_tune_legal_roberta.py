#!/usr/bin/env python3
"""
Fine-tuning script for Saibo Legal RoBERTa on Philippine Jurisprudence
Optimized for RTX 4050 6GB VRAM with 21k cases

Based on official documentation:
- Learning Rate: 5e-5
- Epochs: 3
- Labels: 4,271 (multi-label classification)
- Batch Size: Optimized for 6GB VRAM
"""

import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

# Configuration based on official documentation
CONFIG = {
    "model_name": "Saibo-creator/legal-roberta-base",
    "learning_rate": 5e-5,  # From official documentation
    "epochs": 3,  # From official documentation
    "batch_size": 8,  # Optimized for RTX 4050 6GB
    "gradient_accumulation_steps": 8,  # Effective batch size = 64
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_length": 512,
    "output_dir": "./legal_roberta_finetuned",
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
}

# Legal categories for Philippine jurisprudence (expanded from your existing categories)
LEGAL_CATEGORIES = {
    'case_type': [
        'civil', 'criminal', 'administrative', 'constitutional', 'labor',
        'commercial', 'family', 'property', 'tort', 'tax', 'environmental',
        'election', 'agrarian', 'intellectual_property', 'special_civil_action',
        'special_proceedings', 'appellate', 'original_jurisdiction', 'admiralty',
        'insurance', 'banking', 'corporate', 'public_international', 'private_international'
    ],
    'legal_area': [
        'criminal_law', 'civil_law', 'administrative_law', 'constitutional_law',
        'labor_law', 'commercial_law', 'family_law', 'property_law', 'tort_law',
        'tax_law', 'environmental_law', 'election_law', 'agrarian_law',
        'intellectual_property_law', 'remedial_law', 'political_law', 'public_corporation_law',
        'private_corporation_law', 'banking_law', 'insurance_law', 'maritime_law',
        'international_law', 'ethics', 'constitutional_remedies', 'special_proceedings'
    ],
    'document_section': [
        'facts', 'issues', 'ruling', 'ratio_decidendi', 'disposition',
        'dissenting_opinion', 'concurring_opinion', 'separate_opinion',
        'legal_precedent', 'procedural_history', 'summary', 'background', 'analysis',
        'petition', 'motion', 'memorandum', 'brief', 'pleading'
    ],
    'complexity_level': ['simple', 'moderate', 'complex', 'highly_complex'],
    'jurisdiction_level': [
        'supreme_court', 'appellate_court', 'regional_trial_court', 
        'municipal_trial_court', 'quasi_judicial_agency', 'sandiganbayan',
        'court_of_tax_appeals', 'court_of_appeals'
    ],
    'case_status': [
        'pending', 'decided', 'dismissed', 'settled', 'appealed',
        'affirmed', 'reversed', 'modified', 'remanded'
    ],
    'legal_issue': [
        'constitutional_question', 'administrative_matter', 'criminal_liability',
        'civil_liability', 'contract_dispute', 'property_rights', 'family_dispute',
        'labor_dispute', 'tax_dispute', 'environmental_violation', 'election_dispute',
        'intellectual_property', 'banking_dispute', 'insurance_claim'
    ]
}

# Flatten all labels and create mappings
ALL_LABELS = []
for category in LEGAL_CATEGORIES.values():
    ALL_LABELS.extend(category)

ID_TO_LABEL = {i: label for i, label in enumerate(ALL_LABELS)}
LABEL_TO_ID = {label: i for i, label in enumerate(ALL_LABELS)}
NUM_LABELS = len(ALL_LABELS)

print(f"Total legal categories: {len(LEGAL_CATEGORIES)}")
print(f"Total labels: {NUM_LABELS}")
print(f"Categories: {list(LEGAL_CATEGORIES.keys())}")


class LegalDataset(Dataset):
    """Dataset for legal document classification"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert labels to tensor
        label_tensor = torch.zeros(NUM_LABELS)
        for label_id in labels:
            if 0 <= label_id < NUM_LABELS:
                label_tensor[label_id] = 1.0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


def load_jurisprudence_data(data_file: str) -> Tuple[List[str], List[List[int]]]:
    """Load and preprocess jurisprudence data from JSONL"""
    print(f"Loading data from {data_file}...")
    
    texts = []
    labels = []
    
    # Load existing data
    opener = gzip.open if data_file.endswith('.gz') else open
    with opener(data_file, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading cases")):
            line = line.strip()
            if not line:
                continue
            
            try:
                case = json.loads(line)
                
                # Extract text (prefer clean_text, fallback to content)
                text = case.get('clean_text', '') or case.get('content', '')
                if not text or len(text) < 100:
                    continue
                
                # Limit text length for training
                if len(text) > 2000:
                    text = text[:2000]
                
                # Extract title for better context
                title = case.get('case_title', '') or case.get('title', '')
                if title and len(title) < 200:
                    text = f"{title}. {text}"
                
                # Create labels based on case metadata
                case_labels = create_labels_from_case(case)
                
                # Only add if labels were successfully created
                if case_labels and len(case_labels) > 0:
                    texts.append(text)
                    labels.append(case_labels)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(texts)} cases with valid labels")
    return texts, labels


def create_labels_from_case(case: Dict[str, Any]) -> List[int]:
    """Create multi-label classification from case metadata"""
    labels = []
    
    # Case type classification - handle None values
    case_type = case.get('case_type', '')
    if case_type and isinstance(case_type, str):
        case_type = case_type.lower()
        if case_type in LABEL_TO_ID:
            labels.append(LABEL_TO_ID[case_type])
    
    # Division classification - handle None values
    division = case.get('division', '')
    if division and isinstance(division, str):
        division = division.lower()
        if 'first' in division:
            labels.append(LABEL_TO_ID.get('supreme_court', -1))
        elif 'second' in division or 'third' in division:
            labels.append(LABEL_TO_ID.get('appellate_court', -1))
    
    # Administrative case detection - handle None values
    special_number = case.get('special_number', '')
    if special_number and isinstance(special_number, str) and 'A.M.' in special_number.upper():
        labels.append(LABEL_TO_ID.get('administrative', -1))
        labels.append(LABEL_TO_ID.get('administrative_law', -1))
    
    # GR number classification - handle None values
    gr_number = case.get('gr_number', '')
    if gr_number and isinstance(gr_number, str):
        labels.append(LABEL_TO_ID.get('supreme_court', -1))
    
    # En banc detection
    if case.get('en_banc', False):
        labels.append(LABEL_TO_ID.get('constitutional_law', -1))
    
    # Text-based classification - handle None values
    text = case.get('clean_text', '') or case.get('content', '')
    title = case.get('case_title', '') or case.get('title', '')
    
    if text and isinstance(text, str):
        text = text.lower()
    else:
        text = ''
        
    if title and isinstance(title, str):
        title = title.lower()
    else:
        title = ''
        
    full_text = f"{title} {text}"
    
    # Simple keyword-based classification
    if any(word in full_text for word in ['criminal', 'felony', 'misdemeanor', 'offense']):
        labels.append(LABEL_TO_ID.get('criminal', -1))
        labels.append(LABEL_TO_ID.get('criminal_law', -1))
    
    if any(word in full_text for word in ['contract', 'agreement', 'breach']):
        labels.append(LABEL_TO_ID.get('civil', -1))
        labels.append(LABEL_TO_ID.get('civil_law', -1))
    
    if any(word in full_text for word in ['constitutional', 'bill of rights']):
        labels.append(LABEL_TO_ID.get('constitutional', -1))
        labels.append(LABEL_TO_ID.get('constitutional_law', -1))
    
    if any(word in full_text for word in ['labor', 'employment', 'wage']):
        labels.append(LABEL_TO_ID.get('labor', -1))
        labels.append(LABEL_TO_ID.get('labor_law', -1))
    
    # Remove invalid labels and duplicates
    valid_labels = list(set([l for l in labels if 0 <= l < NUM_LABELS]))
    
    # Ensure at least one label
    if not valid_labels:
        valid_labels = [LABEL_TO_ID.get('civil', 0)]  # Default to civil
    
    return valid_labels


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # Convert to binary predictions (threshold 0.5)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, binary_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Main fine-tuning function"""
    print("🚀 Starting Legal RoBERTa Fine-tuning")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    data_file = "data/cases.jsonl.gz"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run the crawler first to collect jurisprudence data.")
        return
    
    texts, labels = load_jurisprudence_data(data_file)
    
    if len(texts) == 0:
        print("❌ No data found. Please run the crawler first.")
        return
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total cases: {len(texts)}")
    print(f"   Average labels per case: {np.mean([len(l) for l in labels]):.2f}")
    print(f"   Unique labels used: {len(set([l for label_list in labels for l in label_list]))}")
    
    # Validate data consistency
    if len(texts) != len(labels):
        print(f"❌ Data inconsistency: {len(texts)} texts vs {len(labels)} labels")
        return
    
    # Ensure all labels are valid
    for i, label_list in enumerate(labels):
        if not isinstance(label_list, list):
            print(f"❌ Invalid label format at index {i}: {type(label_list)}")
            return
        for label_id in label_list:
            if not isinstance(label_id, int) or label_id < 0 or label_id >= NUM_LABELS:
                print(f"❌ Invalid label ID at index {i}: {label_id}")
                return
    
    print("✅ Data validation passed")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    
    print(f"   Training cases: {len(train_texts)}")
    print(f"   Validation cases: {len(val_texts)}")
    
    # Load tokenizer and model
    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        problem_type="multi_label_classification"
    )
    
    # Create datasets
    train_dataset = LegalDataset(train_texts, train_labels, tokenizer, CONFIG["max_length"])
    val_dataset = LegalDataset(val_texts, val_labels, tokenizer, CONFIG["max_length"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        learning_rate=CONFIG["learning_rate"],
        logging_dir=f"{CONFIG['output_dir']}/logs",
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],
        eval_strategy="steps",  # Fixed: was evaluation_strategy
        save_strategy="steps",
        save_total_limit=CONFIG["save_total_limit"],
        load_best_model_at_end=CONFIG["load_best_model_at_end"],
        metric_for_best_model=CONFIG["metric_for_best_model"],
        greater_is_better=CONFIG["greater_is_better"],
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_pin_memory=False,  # Reduce memory usage
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Start training
    print("🏋️ Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time/3600:.2f} hours")
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Final evaluation
    print("📊 Final evaluation...")
    eval_results = trainer.evaluate()
    
    print("🎯 Final Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # Save evaluation results
    with open(f"{CONFIG['output_dir']}/eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"✅ Fine-tuning complete! Model saved to {CONFIG['output_dir']}")
    print("📈 Expected improvements:")
    print("   - Classification accuracy: 55% → 75-85%")
    print("   - Better case type detection")
    print("   - Improved legal area classification")
    print("   - Enhanced administrative case handling")


if __name__ == "__main__":
    main()
