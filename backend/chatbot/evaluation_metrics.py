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
    def generate_automated_ground_truth(responses: List[str], 
                                      reference_texts: List[str],
                                      case_metadata_list: List[Dict],
                                      query_list: Optional[List[str]] = None) -> List[int]:
        """
        Generate automated ground truth labels when no expert/user ratings are available
        
        Args:
            responses: List of chatbot responses to evaluate
            reference_texts: List of reference legal texts from JSONL
            case_metadata_list: List of case metadata dictionaries
            query_list: Optional list of user queries
            
        Returns:
            List of binary ground truth labels (1=high quality, 0=low quality)
        """
        ground_truth = []
        
        for i, (response, reference, metadata) in enumerate(zip(responses, reference_texts, case_metadata_list)):
            query = query_list[i] if query_list else ""
            
            # Calculate comprehensive automated score
            automated_scores = AutomatedContentScoring.score_legal_response(
                response, reference, metadata
            )
            
            # Generate ground truth using multiple criteria
            gt_score = LegalAccuracyMetrics._calculate_automated_ground_truth_score(
                response, reference, metadata, query, automated_scores
            )
            
            # Binary classification: 1 if high quality, 0 if low quality
            # Adjusted threshold based on demo results
            ground_truth.append(1 if gt_score >= 0.5 else 0)
        
        return ground_truth
    
    @staticmethod
    def _calculate_automated_ground_truth_score(response: str, reference_text: str, 
                                             case_metadata: Dict, query: str, 
                                             automated_scores: Dict[str, Any]) -> float:
        """
        Calculate automated ground truth score using multiple criteria
        
        Returns:
            Float score between 0.0 and 1.0 representing response quality
        """
        scores = []
        weights = []
        
        # 1. Content Relevance (BLEU + ROUGE)
        bleu_avg = automated_scores['bleu'].get('bleu_avg', 0.0)
        rouge_1_f1 = automated_scores['rouge']['rouge_1']['f1']
        rouge_2_f1 = automated_scores['rouge']['rouge_2']['f1']
        rouge_l_f1 = automated_scores['rouge']['rouge_l']['f1']
        
        content_score = (bleu_avg + rouge_1_f1 + rouge_2_f1 + rouge_l_f1) / 4.0
        scores.append(content_score)
        weights.append(0.25)
        
        # 2. Legal Element Presence
        legal_elements_score = automated_scores['legal_elements']['presence_rate']
        scores.append(legal_elements_score)
        weights.append(0.30)
        
        # 3. Citation Accuracy
        citation_score = automated_scores['citation_accuracy']['f1']
        scores.append(citation_score)
        weights.append(0.20)
        
        # 4. Response Completeness (length and structure)
        completeness_score = LegalAccuracyMetrics._calculate_completeness_score(response, reference_text)
        scores.append(completeness_score)
        weights.append(0.15)
        
        # 5. Query Relevance (if query provided)
        if query:
            relevance_score = LegalAccuracyMetrics._calculate_query_relevance_score(query, response, reference_text)
            scores.append(relevance_score)
            weights.append(0.10)
        
        # Calculate weighted average
        total_weight = sum(weights)
        weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
        
        return min(max(weighted_score, 0.0), 1.0)  # Clamp between 0 and 1
    
    @staticmethod
    def _calculate_completeness_score(response: str, reference_text: str) -> float:
        """Calculate response completeness score"""
        if not response or not reference_text:
            return 0.0
        
        # Length ratio (response should be substantial but not too long)
        length_ratio = len(response) / len(reference_text) if len(reference_text) > 0 else 0
        optimal_ratio = 0.3  # Response should be ~30% of reference length
        length_score = 1.0 - abs(length_ratio - optimal_ratio) / optimal_ratio
        length_score = max(0.0, min(1.0, length_score))
        
        # Structure score (check for legal document structure)
        structure_indicators = [
            'case', 'court', 'decision', 'ruling', 'facts', 'issues',
            'held', 'wherefore', 'ordered', 'petitioner', 'respondent'
        ]
        structure_count = sum(1 for indicator in structure_indicators if indicator.lower() in response.lower())
        structure_score = min(structure_count / len(structure_indicators), 1.0)
        
        return (length_score + structure_score) / 2.0
    
    @staticmethod
    def _calculate_query_relevance_score(query: str, response: str, reference_text: str) -> float:
        """Calculate how well the response addresses the query"""
        if not query:
            return 0.5  # Neutral score if no query provided
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        query_terms = {term for term in query_terms if len(term) > 3}  # Remove short words
        
        if not query_terms:
            return 0.5
        
        # Check how many query terms appear in response
        response_terms = set(re.findall(r'\b\w+\b', response_lower))
        matching_terms = query_terms.intersection(response_terms)
        
        # Calculate relevance score
        relevance_score = len(matching_terms) / len(query_terms)
        
        # Bonus for legal terms that might be paraphrased
        legal_synonyms = {
            'case': ['decision', 'ruling', 'judgment'],
            'court': ['tribunal', 'judiciary'],
            'law': ['legal', 'statute', 'regulation'],
            'decision': ['ruling', 'judgment', 'holding']
        }
        
        synonym_matches = 0
        for query_term in query_terms:
            if query_term in legal_synonyms:
                for synonym in legal_synonyms[query_term]:
                    if synonym in response_lower:
                        synonym_matches += 0.5
                        break
        
        relevance_score += min(synonym_matches / len(query_terms), 0.3)  # Cap bonus at 30%
        
        return min(relevance_score, 1.0)
    
    @staticmethod
    def calculate_automated_metrics(responses: List[str], 
                                  reference_texts: List[str],
                                  case_metadata_list: List[Dict],
                                  query_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate metrics using fully automated ground truth generation
        
        Args:
            responses: List of chatbot responses
            reference_texts: List of reference legal texts
            case_metadata_list: List of case metadata
            query_list: Optional list of user queries
            
        Returns:
            Dictionary with automated metrics and ground truth analysis
        """
        # Generate automated ground truth
        ground_truth = LegalAccuracyMetrics.generate_automated_ground_truth(
            responses, reference_texts, case_metadata_list, query_list
        )
        
        # For automated evaluation, we'll use the ground truth as both prediction and truth
        # In practice, you might have separate model predictions
        predictions = ground_truth.copy()
        
        # Calculate standard metrics
        metrics = LegalAccuracyMetrics.calculate_metrics(predictions, ground_truth)
        
        # Calculate additional automated metrics
        automated_metrics = {
            'standard_metrics': metrics,
            'ground_truth_analysis': {
                'total_responses': len(responses),
                'high_quality_count': sum(ground_truth),
                'low_quality_count': len(ground_truth) - sum(ground_truth),
                'quality_distribution': {
                    'high_quality_ratio': sum(ground_truth) / len(ground_truth) if ground_truth else 0,
                    'low_quality_ratio': (len(ground_truth) - sum(ground_truth)) / len(ground_truth) if ground_truth else 0
                }
            },
            'automated_scores_summary': LegalAccuracyMetrics._summarize_automated_scores(
                responses, reference_texts, case_metadata_list
            )
        }
        
        return automated_metrics
    
    @staticmethod
    def _summarize_automated_scores(responses: List[str], reference_texts: List[str], 
                                  case_metadata_list: List[Dict]) -> Dict[str, float]:
        """Summarize automated scores across all responses"""
        all_scores = {
            'bleu_avg': [],
            'rouge_1_f1': [],
            'rouge_2_f1': [],
            'rouge_l_f1': [],
            'legal_elements_presence_rate': [],
            'citation_accuracy_f1': [],
            'overall_relevance': []
        }
        
        for response, reference, metadata in zip(responses, reference_texts, case_metadata_list):
            scores = AutomatedContentScoring.score_legal_response(response, reference, metadata)
            
            all_scores['bleu_avg'].append(scores['bleu']['bleu_avg'])
            all_scores['rouge_1_f1'].append(scores['rouge']['rouge_1']['f1'])
            all_scores['rouge_2_f1'].append(scores['rouge']['rouge_2']['f1'])
            all_scores['rouge_l_f1'].append(scores['rouge']['rouge_l']['f1'])
            all_scores['legal_elements_presence_rate'].append(scores['legal_elements']['presence_rate'])
            all_scores['citation_accuracy_f1'].append(scores['citation_accuracy']['f1'])
            all_scores['overall_relevance'].append(scores['overall_relevance']['score'])
        
        # Calculate averages
        summary = {}
        for metric, values in all_scores.items():
            if values:
                summary[f'avg_{metric}'] = round(np.mean(values), 4)
                summary[f'std_{metric}'] = round(np.std(values), 4)
                summary[f'min_{metric}'] = round(min(values), 4)
                summary[f'max_{metric}'] = round(max(values), 4)
        
        return summary
    
    @staticmethod
    def assess_legal_information_accuracy(response: str, reference_text: str, 
                                        case_metadata: Dict, query: str = "") -> Dict[str, Any]:
        """
        Assess legal information accuracy using automated ground truth
        
        Args:
            response: Chatbot response to assess
            reference_text: Reference legal text
            case_metadata: Case metadata
            query: User query (optional)
            
        Returns:
            Dictionary with accuracy assessment and recommendations
        """
        # Calculate automated scores
        automated_scores = AutomatedContentScoring.score_legal_response(
            response, reference_text, case_metadata
        )
        
        # Generate ground truth score
        gt_score = LegalAccuracyMetrics._calculate_automated_ground_truth_score(
            response, reference_text, case_metadata, query, automated_scores
        )
        
        # Determine accuracy level (adjusted thresholds)
        if gt_score >= 0.7:
            accuracy_level = "HIGH"
            recommendation = "Response demonstrates high legal information accuracy"
        elif gt_score >= 0.4:
            accuracy_level = "MEDIUM"
            recommendation = "Response shows moderate legal information accuracy, consider improvements"
        else:
            accuracy_level = "LOW"
            recommendation = "Response has low legal information accuracy, significant improvements needed"
        
        # Identify specific accuracy issues
        accuracy_issues = []
        if automated_scores['citation_accuracy']['f1'] < 0.7:
            accuracy_issues.append("Citation accuracy is low - verify G.R. numbers and case references")
        if automated_scores['legal_elements']['presence_rate'] < 0.6:
            accuracy_issues.append("Missing key legal elements - ensure case details are complete")
        if automated_scores['bleu']['bleu_avg'] < 0.3:
            accuracy_issues.append("Content relevance is low - response may not match reference accurately")
        if automated_scores['rouge']['rouge_1']['f1'] < 0.4:
            accuracy_issues.append("Information overlap is insufficient - ensure key facts are included")
        
        return {
            'accuracy_level': accuracy_level,
            'accuracy_score': round(gt_score, 4),
            'recommendation': recommendation,
            'accuracy_issues': accuracy_issues,
            'detailed_scores': {
                'content_relevance': round(automated_scores['bleu']['bleu_avg'], 4),
                'information_overlap': round(automated_scores['rouge']['rouge_1']['f1'], 4),
                'legal_elements_completeness': round(automated_scores['legal_elements']['presence_rate'], 4),
                'citation_accuracy': round(automated_scores['citation_accuracy']['f1'], 4),
                'overall_relevance': round(automated_scores['overall_relevance']['score'], 4)
            },
            'quality_threshold_met': gt_score >= 0.7
        }
    
    @staticmethod
    def monitor_legal_accuracy_trends(evaluation_logs: List[Dict]) -> Dict[str, Any]:
        """
        Monitor legal accuracy trends over time
        
        Args:
            evaluation_logs: List of evaluation records
            
        Returns:
            Dictionary with accuracy trends and insights
        """
        if not evaluation_logs:
            return {'error': 'No evaluation logs provided'}
        
        # Extract accuracy scores
        accuracy_scores = []
        timestamps = []
        
        for log in evaluation_logs:
            if 'automated_ground_truth_score' in log:
                accuracy_scores.append(log['automated_ground_truth_score'])
                timestamps.append(log.get('timestamp', ''))
        
        if not accuracy_scores:
            return {'error': 'No accuracy scores found in logs'}
        
        # Calculate trends
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        min_accuracy = min(accuracy_scores)
        max_accuracy = max(accuracy_scores)
        
        # Count accuracy levels
        high_accuracy_count = sum(1 for score in accuracy_scores if score >= 0.8)
        medium_accuracy_count = sum(1 for score in accuracy_scores if 0.6 <= score < 0.8)
        low_accuracy_count = sum(1 for score in accuracy_scores if score < 0.6)
        
        # Calculate improvement trend (if multiple evaluations)
        trend = "stable"
        if len(accuracy_scores) > 1:
            recent_avg = sum(accuracy_scores[-5:]) / min(5, len(accuracy_scores))
            earlier_avg = sum(accuracy_scores[:5]) / min(5, len(accuracy_scores))
            
            if recent_avg > earlier_avg + 0.05:
                trend = "improving"
            elif recent_avg < earlier_avg - 0.05:
                trend = "declining"
        
        return {
            'total_evaluations': len(accuracy_scores),
            'average_accuracy': round(avg_accuracy, 4),
            'min_accuracy': round(min_accuracy, 4),
            'max_accuracy': round(max_accuracy, 4),
            'accuracy_distribution': {
                'high_accuracy': high_accuracy_count,
                'medium_accuracy': medium_accuracy_count,
                'low_accuracy': low_accuracy_count,
                'high_accuracy_percentage': round(high_accuracy_count / len(accuracy_scores) * 100, 1)
            },
            'trend': trend,
            'recommendations': LegalAccuracyMetrics._generate_accuracy_recommendations(
                avg_accuracy, trend, high_accuracy_count, len(accuracy_scores)
            )
        }
    
    @staticmethod
    def _generate_accuracy_recommendations(avg_accuracy: float, trend: str, 
                                         high_accuracy_count: int, total_count: int) -> List[str]:
        """Generate recommendations based on accuracy analysis"""
        recommendations = []
        
        if avg_accuracy < 0.6:
            recommendations.append("Overall accuracy is low - focus on improving legal element detection")
            recommendations.append("Review citation accuracy validation algorithms")
        elif avg_accuracy < 0.8:
            recommendations.append("Accuracy is moderate - consider enhancing content relevance scoring")
            recommendations.append("Improve legal element completeness validation")
        
        if trend == "declining":
            recommendations.append("Accuracy trend is declining - investigate recent changes")
        elif trend == "improving":
            recommendations.append("Accuracy is improving - continue current strategies")
        
        high_accuracy_rate = high_accuracy_count / total_count
        if high_accuracy_rate < 0.5:
            recommendations.append("Less than 50% of responses achieve high accuracy - review quality thresholds")
        
        return recommendations

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
        
        # Check content sections (more lenient for shorter responses)
        section_keywords = {
            'facts': ['fact', 'background', 'petitioner', 'respondent', 'alleged', 'case', 'court'],
            'issues': ['issue', 'question', 'whether', 'contention', 'argument', 'problem'],
            'ruling': ['ruling', 'held', 'decision', 'ordered', 'wherefore', 'decided', 'found', 'guilty', 'innocent'],
            'legal_doctrine': ['doctrine', 'principle', 'rule', 'test', 'standard', 'law', 'legal']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                elements_present[section] = True
        
        # Bonus for legal terminology presence
        legal_terms = ['supreme court', 'court', 'case', 'law', 'legal', 'defendant', 'plaintiff', 'guilty', 'innocent', 'ruling', 'decision']
        legal_term_count = sum(1 for term in legal_terms if term in response_lower)
        if legal_term_count >= 2:  # If response contains at least 2 legal terms
            elements_present['legal_doctrine'] = True
        
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
    
    def log_evaluation_with_automated_ground_truth(self, query: str, response: str, reference: str, 
                                                  case_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Log evaluation using automated ground truth generation (no expert/user ratings needed)
        
        Args:
            query: User query
            response: Chatbot response
            reference: Reference text
            case_metadata: Case metadata
            
        Returns:
            Complete evaluation record with automated ground truth
        """
        # Calculate automated scores
        automated_scores = AutomatedContentScoring.score_legal_response(
            response, reference, case_metadata
        )
        
        # Generate automated ground truth score
        gt_score = LegalAccuracyMetrics._calculate_automated_ground_truth_score(
            response, reference, case_metadata or {}, query, automated_scores
        )
        
        # Create automated expert scores for compatibility
        automated_expert_scores = {
            'accuracy': round(gt_score * 5, 1),  # Scale to 1-5 range
            'completeness': round(automated_scores['legal_elements']['presence_rate'] * 5, 1),
            'relevance': round(automated_scores['overall_relevance']['score'] * 5, 1),
            'clarity': round(min(gt_score * 5, 5.0), 1),
            'legal_reasoning': round(automated_scores['citation_accuracy']['f1'] * 5, 1),
            'citation_accuracy': round(automated_scores['citation_accuracy']['f1'] * 5, 1),
            'overall_rating': round(gt_score * 5, 1)
        }
        
        # Create evaluation record
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'query': query,
            'response': response,
            'reference': reference[:500],  # Store first 500 chars
            'case_metadata': case_metadata,
            'automated_scores': automated_scores,
            'expert_scores': automated_expert_scores,
            'automated_ground_truth_score': round(gt_score, 4),
            'quality_label': 'high_quality' if gt_score >= 0.7 else 'low_quality',
            'response_length': len(response),
            'reference_length': len(reference)
        }
        
        # Save to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evaluation_record, ensure_ascii=False) + '\n')
        
        return evaluation_record
    
    def export_report(self, output_file: Optional[str] = None) -> str:
        """Export evaluation report to JSON file"""
        if output_file is None:
            output_file = os.path.join(self.log_dir, f"report_{self.session_id}.json")
        
        statistics = self.get_session_statistics()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        return output_file
