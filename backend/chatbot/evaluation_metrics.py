# evaluation_metrics.py — Comprehensive evaluation metrics for legal chatbot
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# LEGAL INFORMATION ACCURACY METRICS
# =============================================================================

class LegalAccuracyMetrics:
    """Classification metrics for legal information accuracy"""
    
    @staticmethod
    def calculate_metrics(predictions: List[int], ground_truth: List[int]) -> Dict[str, float]:
        """
        Calculate accuracy, precision, recall, F1 score, and specificity
        
        Args:
            predictions: List of predicted labels (0 or 1)
            ground_truth: List of true labels (0 or 1)
            
        Returns:
            Dictionary containing all classification metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        if not predictions or not ground_truth:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0
            }
        
        # Calculate confusion matrix components
        true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
        true_negatives = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 0)
        false_positives = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 0)
        false_negatives = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 1)
        
        total = len(predictions)
        
        # Accuracy: (TP + TN) / Total
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        
        # Precision: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity: TN / (TN + FP)
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'specificity': round(specificity, 4),
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    @staticmethod
    def calculate_multiclass_metrics(predictions: List[int], ground_truth: List[int], num_classes: int) -> Dict[str, Any]:
        """
        Calculate metrics for multiclass classification
        
        Args:
            predictions: List of predicted class labels
            ground_truth: List of true class labels
            num_classes: Number of classes
            
        Returns:
            Dictionary containing per-class and macro-averaged metrics
        """
        per_class_metrics = {}
        
        for class_label in range(num_classes):
            # Convert to binary classification for this class
            binary_predictions = [1 if p == class_label else 0 for p in predictions]
            binary_ground_truth = [1 if g == class_label else 0 for g in ground_truth]
            
            metrics = LegalAccuracyMetrics.calculate_metrics(binary_predictions, binary_ground_truth)
            per_class_metrics[f'class_{class_label}'] = metrics
        
        # Calculate macro-averaged metrics
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
        
        return {
            'per_class': per_class_metrics,
            'macro_avg': {
                'precision': round(macro_precision, 4),
                'recall': round(macro_recall, 4),
                'f1_score': round(macro_f1, 4)
            }
        }


# =============================================================================
# CONTENT RELEVANCE METRICS (BLEU & ROUGE)
# =============================================================================

class ContentRelevanceMetrics:
    """BLEU and ROUGE metrics for content relevance evaluation"""
    
    @staticmethod
    def calculate_bleu(candidate: str, references: List[str], max_n: int = 4) -> Dict[str, float]:
        """
        Calculate BLEU score for candidate text against reference texts
        
        Args:
            candidate: Generated text from chatbot
            references: List of reference legal texts
            max_n: Maximum n-gram size (default 4)
            
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        # Tokenize
        candidate_tokens = ContentRelevanceMetrics._tokenize(candidate)
        reference_tokens_list = [ContentRelevanceMetrics._tokenize(ref) for ref in references]
        
        if not candidate_tokens or not any(reference_tokens_list):
            return {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)}
        
        bleu_scores = {}
        
        for n in range(1, max_n + 1):
            # Calculate n-gram precision
            candidate_ngrams = ContentRelevanceMetrics._get_ngrams(candidate_tokens, n)
            
            if not candidate_ngrams:
                bleu_scores[f'bleu_{n}'] = 0.0
                continue
            
            max_overlap = 0
            for reference_tokens in reference_tokens_list:
                reference_ngrams = ContentRelevanceMetrics._get_ngrams(reference_tokens, n)
                overlap = sum((candidate_ngrams & reference_ngrams).values())
                max_overlap = max(max_overlap, overlap)
            
            precision = max_overlap / sum(candidate_ngrams.values()) if sum(candidate_ngrams.values()) > 0 else 0.0
            bleu_scores[f'bleu_{n}'] = round(precision, 4)
        
        # Calculate geometric mean (BLEU score)
        bleu_avg = np.exp(np.mean([np.log(score + 1e-10) for score in bleu_scores.values()]))
        bleu_scores['bleu_avg'] = round(bleu_avg, 4)
        
        return bleu_scores
    
    @staticmethod
    def calculate_rouge(candidate: str, references: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for candidate text
        
        Args:
            candidate: Generated text from chatbot
            references: List of reference legal texts
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores (precision, recall, F1)
        """
        candidate_tokens = ContentRelevanceMetrics._tokenize(candidate)
        reference_tokens_list = [ContentRelevanceMetrics._tokenize(ref) for ref in references]
        
        if not candidate_tokens or not any(reference_tokens_list):
            return {
                'rouge_1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'rouge_2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'rouge_l': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            }
        
        rouge_scores = {}
        
        # ROUGE-1 (unigram overlap)
        rouge_scores['rouge_1'] = ContentRelevanceMetrics._calculate_rouge_n(
            candidate_tokens, reference_tokens_list, n=1
        )
        
        # ROUGE-2 (bigram overlap)
        rouge_scores['rouge_2'] = ContentRelevanceMetrics._calculate_rouge_n(
            candidate_tokens, reference_tokens_list, n=2
        )
        
        # ROUGE-L (longest common subsequence)
        rouge_scores['rouge_l'] = ContentRelevanceMetrics._calculate_rouge_l(
            candidate_tokens, reference_tokens_list
        )
        
        return rouge_scores
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into words, handling legal text specifics"""
        # Convert to lowercase
        text = text.lower()
        
        # Keep legal citations intact (e.g., G.R. No. 123456)
        text = re.sub(r'g\.r\.\s*no\.\s*\d+', lambda m: m.group(0).replace(' ', '_'), text)
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    @staticmethod
    def _calculate_rouge_n(candidate_tokens: List[str], reference_tokens_list: List[List[str]], n: int) -> Dict[str, float]:
        """Calculate ROUGE-N scores"""
        candidate_ngrams = ContentRelevanceMetrics._get_ngrams(candidate_tokens, n)
        
        if not candidate_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        max_precision = 0.0
        max_recall = 0.0
        max_f1 = 0.0
        
        for reference_tokens in reference_tokens_list:
            reference_ngrams = ContentRelevanceMetrics._get_ngrams(reference_tokens, n)
            
            if not reference_ngrams:
                continue
            
            overlap = sum((candidate_ngrams & reference_ngrams).values())
            
            precision = overlap / sum(candidate_ngrams.values()) if sum(candidate_ngrams.values()) > 0 else 0.0
            recall = overlap / sum(reference_ngrams.values()) if sum(reference_ngrams.values()) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            max_precision = max(max_precision, precision)
            max_recall = max(max_recall, recall)
            max_f1 = max(max_f1, f1)
        
        return {
            'precision': round(max_precision, 4),
            'recall': round(max_recall, 4),
            'f1': round(max_f1, 4)
        }
    
    @staticmethod
    def _calculate_rouge_l(candidate_tokens: List[str], reference_tokens_list: List[List[str]]) -> Dict[str, float]:
        """Calculate ROUGE-L (longest common subsequence)"""
        max_precision = 0.0
        max_recall = 0.0
        max_f1 = 0.0
        
        for reference_tokens in reference_tokens_list:
            lcs_length = ContentRelevanceMetrics._lcs_length(candidate_tokens, reference_tokens)
            
            precision = lcs_length / len(candidate_tokens) if len(candidate_tokens) > 0 else 0.0
            recall = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            max_precision = max(max_precision, precision)
            max_recall = max(max_recall, recall)
            max_f1 = max(max_f1, f1)
        
        return {
            'precision': round(max_precision, 4),
            'recall': round(max_recall, 4),
            'f1': round(max_f1, 4)
        }
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]


# =============================================================================
# AUTOMATED CONTENT SCORING
# =============================================================================

class AutomatedContentScoring:
    """Automated scoring system for content relevance"""
    
    @staticmethod
    def score_legal_response(response: str, reference_text: str, case_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive automated scoring of chatbot response
        
        Args:
            response: Chatbot generated response
            reference_text: Reference legal text (from JSONL)
            case_metadata: Optional case metadata for validation
            
        Returns:
            Dictionary with multiple relevance scores
        """
        scores = {}
        
        # 1. BLEU scores
        bleu_scores = ContentRelevanceMetrics.calculate_bleu(response, [reference_text])
        scores['bleu'] = bleu_scores
        
        # 2. ROUGE scores
        rouge_scores = ContentRelevanceMetrics.calculate_rouge(response, [reference_text])
        scores['rouge'] = rouge_scores
        
        # 3. Legal element presence
        legal_elements = AutomatedContentScoring._check_legal_elements(response, case_metadata)
        scores['legal_elements'] = legal_elements
        
        # 4. Citation accuracy
        citation_accuracy = AutomatedContentScoring._check_citations(response, reference_text)
        scores['citation_accuracy'] = citation_accuracy
        
        # 5. Overall content relevance score (weighted average)
        overall_score = AutomatedContentScoring._calculate_overall_score(scores)
        scores['overall_relevance'] = overall_score
        
        return scores
    
    @staticmethod
    def _check_legal_elements(response: str, case_metadata: Optional[Dict]) -> Dict[str, Any]:
        """Check presence of key legal elements"""
        elements_present = {
            'case_title': False,
            'gr_number': False,
            'ponente': False,
            'date': False,
            'case_type': False,
            'facts': False,
            'issues': False,
            'ruling': False,
            'legal_doctrine': False
        }
        
        response_lower = response.lower()
        
        # Check metadata elements
        if case_metadata:
            if case_metadata.get('case_title', '') and case_metadata['case_title'].lower() in response_lower:
                elements_present['case_title'] = True
            
            if case_metadata.get('gr_number', '') and case_metadata['gr_number'] in response:
                elements_present['gr_number'] = True
            
            if case_metadata.get('ponente', '') and case_metadata['ponente'].lower() in response_lower:
                elements_present['ponente'] = True
            
            if case_metadata.get('promulgation_date', ''):
                date_str = str(case_metadata['promulgation_date'])
                if date_str in response or date_str[:4] in response:  # Year at minimum
                    elements_present['date'] = True
            
            if case_metadata.get('case_type', '') and case_metadata['case_type'].lower() in response_lower:
                elements_present['case_type'] = True
        
        # Check content sections
        section_keywords = {
            'facts': ['fact', 'background', 'petitioner', 'respondent', 'alleged'],
            'issues': ['issue', 'question', 'whether', 'contention', 'argument'],
            'ruling': ['ruling', 'held', 'decision', 'ordered', 'wherefore', 'decided'],
            'legal_doctrine': ['doctrine', 'principle', 'rule', 'test', 'standard']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                elements_present[section] = True
        
        # Calculate presence rate
        presence_rate = sum(elements_present.values()) / len(elements_present)
        
        return {
            'elements': elements_present,
            'presence_rate': round(presence_rate, 4),
            'elements_found': sum(elements_present.values()),
            'total_elements': len(elements_present)
        }
    
    @staticmethod
    def _check_citations(response: str, reference_text: str) -> Dict[str, float]:
        """Check accuracy of legal citations"""
        # Extract G.R. numbers from both texts
        response_citations = set(re.findall(r'G\.R\.\s*No\.\s*\d+', response, re.IGNORECASE))
        reference_citations = set(re.findall(r'G\.R\.\s*No\.\s*\d+', reference_text, re.IGNORECASE))
        
        if not response_citations:
            return {
                'precision': 1.0 if not reference_citations else 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Calculate citation overlap
        correct_citations = response_citations & reference_citations
        
        precision = len(correct_citations) / len(response_citations) if response_citations else 0.0
        recall = len(correct_citations) / len(reference_citations) if reference_citations else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
    
    @staticmethod
    def _calculate_overall_score(scores: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall content relevance score using weighted average"""
        weights = {
            'bleu': 0.25,
            'rouge': 0.25,
            'legal_elements': 0.30,
            'citation_accuracy': 0.20
        }
        
        # Extract component scores
        bleu_score = scores['bleu'].get('bleu_avg', 0.0)
        rouge_score = np.mean([
            scores['rouge']['rouge_1']['f1'],
            scores['rouge']['rouge_2']['f1'],
            scores['rouge']['rouge_l']['f1']
        ])
        legal_elements_score = scores['legal_elements']['presence_rate']
        citation_score = scores['citation_accuracy']['f1']
        
        # Weighted average
        overall = (
            bleu_score * weights['bleu'] +
            rouge_score * weights['rouge'] +
            legal_elements_score * weights['legal_elements'] +
            citation_score * weights['citation_accuracy']
        )
        
        return {
            'score': round(overall, 4),
            'bleu_component': round(bleu_score, 4),
            'rouge_component': round(rouge_score, 4),
            'legal_elements_component': round(legal_elements_score, 4),
            'citation_component': round(citation_score, 4)
        }


# =============================================================================
# EVALUATION TRACKING AND LOGGING
# =============================================================================

class EvaluationTracker:
    """Track and log evaluation metrics over time"""
    
    def __init__(self, log_dir: str = "backend/data/evaluation_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"evaluation_{self.session_id}.jsonl")
    
    def log_evaluation(self, query: str, response: str, reference: str, 
                      case_metadata: Optional[Dict] = None, 
                      expert_scores: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Log a single evaluation instance
        
        Args:
            query: User query
            response: Chatbot response
            reference: Reference text
            case_metadata: Case metadata
            expert_scores: Optional expert evaluation scores
            
        Returns:
            Complete evaluation record
        """
        # Calculate automated scores
        automated_scores = AutomatedContentScoring.score_legal_response(
            response, reference, case_metadata
        )
        
        # Create evaluation record
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'query': query,
            'response': response,
            'reference': reference[:500],  # Store first 500 chars
            'case_metadata': case_metadata,
            'automated_scores': automated_scores,
            'expert_scores': expert_scores,
            'response_length': len(response),
            'reference_length': len(reference)
        }
        
        # Save to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evaluation_record, ensure_ascii=False) + '\n')
        
        return evaluation_record
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics for current session"""
        if not os.path.exists(self.log_file):
            return {}
        
        evaluations = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                evaluations.append(json.loads(line))
        
        if not evaluations:
            return {}
        
        # Aggregate metrics
        bleu_scores = [e['automated_scores']['bleu']['bleu_avg'] for e in evaluations]
        rouge_1_f1 = [e['automated_scores']['rouge']['rouge_1']['f1'] for e in evaluations]
        rouge_2_f1 = [e['automated_scores']['rouge']['rouge_2']['f1'] for e in evaluations]
        rouge_l_f1 = [e['automated_scores']['rouge']['rouge_l']['f1'] for e in evaluations]
        overall_scores = [e['automated_scores']['overall_relevance']['score'] for e in evaluations]
        legal_element_rates = [e['automated_scores']['legal_elements']['presence_rate'] for e in evaluations]
        
        statistics = {
            'session_id': self.session_id,
            'total_evaluations': len(evaluations),
            'average_scores': {
                'bleu_avg': round(np.mean(bleu_scores), 4),
                'rouge_1_f1': round(np.mean(rouge_1_f1), 4),
                'rouge_2_f1': round(np.mean(rouge_2_f1), 4),
                'rouge_l_f1': round(np.mean(rouge_l_f1), 4),
                'overall_relevance': round(np.mean(overall_scores), 4),
                'legal_element_presence': round(np.mean(legal_element_rates), 4)
            },
            'std_scores': {
                'bleu_avg': round(np.std(bleu_scores), 4),
                'rouge_1_f1': round(np.std(rouge_1_f1), 4),
                'rouge_2_f1': round(np.std(rouge_2_f1), 4),
                'rouge_l_f1': round(np.std(rouge_l_f1), 4),
                'overall_relevance': round(np.std(overall_scores), 4)
            }
        }
        
        return statistics
    
    def export_report(self, output_file: Optional[str] = None) -> str:
        """Export evaluation report to JSON file"""
        if output_file is None:
            output_file = os.path.join(self.log_dir, f"report_{self.session_id}.json")
        
        statistics = self.get_session_statistics()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dumps(statistics, indent=2, ensure_ascii=False)
        
        return output_file
