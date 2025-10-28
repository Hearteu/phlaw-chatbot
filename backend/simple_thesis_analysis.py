#!/usr/bin/env python3
"""
Simple Thesis Metrics Analysis - Focus on per-query scores for presentation
"""

import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class SimpleThesisAnalyzer:
    """Simple analyzer focused on per-query scores for presentation"""
    
    def __init__(self):
        self.test_results_dir = "test_results"
        self.evaluation_logs_dir = "data/evaluation_logs"
        self.output_dir = "thesis_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_evaluation_data(self) -> List[Dict]:
        """Load evaluation data from JSONL files"""
        evaluations = []
        
        if not os.path.exists(self.evaluation_logs_dir):
            print(f"Evaluation logs directory not found: {self.evaluation_logs_dir}")
            return evaluations
        
        for filename in os.listdir(self.evaluation_logs_dir):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.evaluation_logs_dir, filename)
                print(f"Loading: {filename}")
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                evaluation = json.loads(line)
                                evaluations.append(evaluation)
                            except json.JSONDecodeError as e:
                                print(f"Error parsing line {line_num} in {filename}: {e}")
        
        print(f"Loaded {len(evaluations)} evaluation records")
        return evaluations
    
    def extract_per_query_scores(self, evaluations: List[Dict]) -> List[Dict[str, Any]]:
        """Extract detailed scores for each individual query"""
        per_query_data = []
        
        for i, evaluation in enumerate(evaluations):
            query_data = {
                'query_id': i + 1,
                'timestamp': evaluation.get('timestamp', ''),
                'session_id': evaluation.get('session_id', ''),
                'query': evaluation.get('query', ''),
                'response': evaluation.get('response', ''),
                'response_length': evaluation.get('response_length', 0),
                'reference_length': evaluation.get('reference_length', 0),
                'quality_label': evaluation.get('quality_label', ''),
                'automated_ground_truth_score': evaluation.get('automated_ground_truth_score', 0.0)
            }
            
            # Extract automated scores if available
            if 'automated_scores' in evaluation:
                automated_scores = evaluation['automated_scores']
                
                # BLEU scores
                if 'bleu' in automated_scores:
                    query_data['bleu_1'] = automated_scores['bleu'].get('bleu_1', 0.0)
                    query_data['bleu_2'] = automated_scores['bleu'].get('bleu_2', 0.0)
                    query_data['bleu_3'] = automated_scores['bleu'].get('bleu_3', 0.0)
                    query_data['bleu_4'] = automated_scores['bleu'].get('bleu_4', 0.0)
                    query_data['bleu_avg'] = automated_scores['bleu'].get('bleu_avg', 0.0)
                else:
                    query_data.update({'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'bleu_avg': 0.0})
                
                # ROUGE scores
                if 'rouge' in automated_scores:
                    query_data['rouge_1_f1'] = automated_scores['rouge'].get('rouge_1', {}).get('f1', 0.0)
                    query_data['rouge_2_f1'] = automated_scores['rouge'].get('rouge_2', {}).get('f1', 0.0)
                    query_data['rouge_l_f1'] = automated_scores['rouge'].get('rouge_l', {}).get('f1', 0.0)
                else:
                    query_data.update({'rouge_1_f1': 0.0, 'rouge_2_f1': 0.0, 'rouge_l_f1': 0.0})
                
                # Legal elements
                if 'legal_elements' in automated_scores:
                    query_data['legal_elements_presence_rate'] = automated_scores['legal_elements'].get('presence_rate', 0.0)
                    query_data['legal_elements_found'] = automated_scores['legal_elements'].get('elements_found', 0)
                    query_data['legal_elements_total'] = automated_scores['legal_elements'].get('total_elements', 0)
                else:
                    query_data.update({'legal_elements_presence_rate': 0.0, 'legal_elements_found': 0, 'legal_elements_total': 0})
                
                # Citation accuracy
                if 'citation_accuracy' in automated_scores:
                    query_data['citation_accuracy_precision'] = automated_scores['citation_accuracy'].get('precision', 0.0)
                    query_data['citation_accuracy_recall'] = automated_scores['citation_accuracy'].get('recall', 0.0)
                    query_data['citation_accuracy_f1'] = automated_scores['citation_accuracy'].get('f1', 0.0)
                else:
                    query_data.update({'citation_accuracy_precision': 0.0, 'citation_accuracy_recall': 0.0, 'citation_accuracy_f1': 0.0})
                
                # Overall relevance
                if 'overall_relevance' in automated_scores:
                    query_data['overall_relevance_score'] = automated_scores['overall_relevance'].get('score', 0.0)
                    query_data['overall_relevance_bleu_component'] = automated_scores['overall_relevance'].get('bleu_component', 0.0)
                    query_data['overall_relevance_rouge_component'] = automated_scores['overall_relevance'].get('rouge_component', 0.0)
                    query_data['overall_relevance_legal_elements_component'] = automated_scores['overall_relevance'].get('legal_elements_component', 0.0)
                    query_data['overall_relevance_citation_component'] = automated_scores['overall_relevance'].get('citation_component', 0.0)
                else:
                    query_data.update({
                        'overall_relevance_score': 0.0, 'overall_relevance_bleu_component': 0.0,
                        'overall_relevance_rouge_component': 0.0, 'overall_relevance_legal_elements_component': 0.0,
                        'overall_relevance_citation_component': 0.0
                    })
            
            # Extract expert scores if available
            if 'expert_scores' in evaluation:
                expert_scores = evaluation['expert_scores']
                query_data['expert_accuracy'] = expert_scores.get('accuracy', 0.0)
                query_data['expert_completeness'] = expert_scores.get('completeness', 0.0)
                query_data['expert_relevance'] = expert_scores.get('relevance', 0.0)
                query_data['expert_clarity'] = expert_scores.get('clarity', 0.0)
                query_data['expert_legal_reasoning'] = expert_scores.get('legal_reasoning', 0.0)
                query_data['expert_citation_accuracy'] = expert_scores.get('citation_accuracy', 0.0)
                query_data['expert_overall_rating'] = expert_scores.get('overall_rating', 0.0)
            else:
                query_data.update({
                    'expert_accuracy': 0.0, 'expert_completeness': 0.0, 'expert_relevance': 0.0,
                    'expert_clarity': 0.0, 'expert_legal_reasoning': 0.0, 'expert_citation_accuracy': 0.0,
                    'expert_overall_rating': 0.0
                })
            
            # Extract case metadata if available
            if 'case_metadata' in evaluation:
                case_metadata = evaluation['case_metadata']
                query_data['case_title'] = case_metadata.get('case_title', '')
                query_data['gr_number'] = case_metadata.get('gr_number', '')
                query_data['ponente'] = case_metadata.get('ponente', '')
                query_data['promulgation_date'] = case_metadata.get('promulgation_date', '')
                query_data['case_type'] = case_metadata.get('case_type', '')
            else:
                query_data.update({
                    'case_title': '', 'gr_number': '', 'ponente': '',
                    'promulgation_date': '', 'case_type': ''
                })
            
            per_query_data.append(query_data)
        
        return per_query_data
    
    def save_csv(self, per_query_scores: List[Dict[str, Any]]) -> str:
        """Save per-query scores as CSV for presentation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"per_query_scores_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        if not per_query_scores:
            print("No per-query scores to save")
            return filepath
        
        # Define CSV headers - include all possible fields
        headers = [
            'query_id', 'timestamp', 'session_id', 'query', 'response', 'response_length', 
            'reference_length', 'quality_label', 'automated_ground_truth_score', 
            'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_avg',
            'rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1',
            'legal_elements_presence_rate', 'legal_elements_found', 'legal_elements_total',
            'citation_accuracy_precision', 'citation_accuracy_recall', 'citation_accuracy_f1',
            'overall_relevance_score', 'overall_relevance_bleu_component', 
            'overall_relevance_rouge_component', 'overall_relevance_legal_elements_component',
            'overall_relevance_citation_component', 'expert_accuracy', 'expert_completeness',
            'expert_relevance', 'expert_clarity', 'expert_legal_reasoning', 'expert_citation_accuracy',
            'expert_overall_rating', 'case_title', 'gr_number', 'ponente', 'promulgation_date', 'case_type'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for query_data in per_query_scores:
                # Truncate long queries for CSV
                if len(query_data.get('query', '')) > 200:
                    query_data['query'] = query_data['query'][:200] + '...'
                
                writer.writerow(query_data)
        
        return filepath
    
    def save_json(self, per_query_scores: List[Dict[str, Any]]) -> str:
        """Save detailed query data as JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_query_data_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        detailed_data = {
            'metadata': {
                'total_queries': len(per_query_scores),
                'generated_at': datetime.now().isoformat(),
                'description': 'Detailed per-query evaluation scores for thesis presentation'
            },
            'queries': per_query_scores
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def calculate_summary_stats(self, per_query_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for thesis objectives"""
        if not per_query_scores:
            return {}
        
        # Extract scores
        bleu_avg_scores = [q.get('bleu_avg', 0.0) for q in per_query_scores]
        rouge_1_scores = [q.get('rouge_1_f1', 0.0) for q in per_query_scores]
        overall_relevance_scores = [q.get('overall_relevance_score', 0.0) for q in per_query_scores]
        ground_truth_scores = [q.get('automated_ground_truth_score', 0.0) for q in per_query_scores]
        
        # Calculate averages
        avg_bleu = sum(bleu_avg_scores) / len(bleu_avg_scores) if bleu_avg_scores else 0.0
        avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0
        avg_overall_relevance = sum(overall_relevance_scores) / len(overall_relevance_scores) if overall_relevance_scores else 0.0
        avg_ground_truth = sum(ground_truth_scores) / len(ground_truth_scores) if ground_truth_scores else 0.0
        
        # Count high quality responses
        high_quality_count = sum(1 for score in ground_truth_scores if score >= 0.7)
        total_queries = len(per_query_scores)
        
        return {
            'total_queries': total_queries,
            'high_quality_queries': high_quality_count,
            'high_quality_percentage': (high_quality_count / total_queries * 100) if total_queries > 0 else 0.0,
            'average_bleu_score': round(avg_bleu, 4),
            'average_rouge_1_score': round(avg_rouge_1, 4),
            'average_overall_relevance': round(avg_overall_relevance, 4),
            'average_ground_truth_score': round(avg_ground_truth, 4),
            'content_relevance_percentage': round(avg_overall_relevance * 100, 2)
        }
    
    def print_summary(self, summary_stats: Dict[str, Any]):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("THESIS METRICS SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Queries Analyzed: {summary_stats.get('total_queries', 0)}")
        print(f"  High Quality Queries: {summary_stats.get('high_quality_queries', 0)}")
        print(f"  High Quality Percentage: {summary_stats.get('high_quality_percentage', 0):.2f}%")
        
        print(f"\n🎯 THESIS OBJECTIVES:")
        print(f"  Average BLEU Score: {summary_stats.get('average_bleu_score', 0):.4f}")
        print(f"  Average ROUGE-1 Score: {summary_stats.get('average_rouge_1_score', 0):.4f}")
        print(f"  Content Relevance Rate: {summary_stats.get('content_relevance_percentage', 0):.2f}%")
        print(f"  Average Ground Truth Score: {summary_stats.get('average_ground_truth_score', 0):.4f}")
        
        # Check thesis objectives
        content_relevance = summary_stats.get('content_relevance_percentage', 0)
        high_quality_rate = summary_stats.get('high_quality_percentage', 0)
        
        print(f"\n✅ THESIS OBJECTIVE STATUS:")
        print(f"  Objective 1 (75-85% Accuracy): {'✓ MET' if 75 <= high_quality_rate <= 85 else '✗ NOT MET'}")
        print(f"  Objective 2 (50-80% Content Relevance): {'✓ MET' if 50 <= content_relevance <= 80 else '✗ NOT MET'}")
        print(f"  Objective 3 (Hallucination Reduction): {'✓ MET' if high_quality_rate >= 70 else '✗ NOT MET'}")
        
        print("\n" + "="*80)

def main():
    """Main function to run simple thesis analysis"""
    analyzer = SimpleThesisAnalyzer()
    
    print("Starting Simple Thesis Metrics Analysis...")
    print("Focus: Per-query scores for presentation\n")
    
    # Load evaluation data
    print("Loading evaluation data...")
    evaluations = analyzer.load_evaluation_data()
    
    if not evaluations:
        print("❌ No evaluation data found!")
        print("Make sure you have evaluation logs in the data/evaluation_logs/ directory")
        return
    
    # Extract per-query scores
    print("Extracting per-query scores...")
    per_query_scores = analyzer.extract_per_query_scores(evaluations)
    
    if not per_query_scores:
        print("❌ No per-query scores extracted!")
        return
    
    # Save CSV file
    print("Saving CSV file...")
    csv_file = analyzer.save_csv(per_query_scores)
    print(f"✅ CSV file saved: {csv_file}")
    
    # Save JSON file
    print("Saving JSON file...")
    json_file = analyzer.save_json(per_query_scores)
    print(f"✅ JSON file saved: {json_file}")
    
    # Calculate and print summary
    print("Calculating summary statistics...")
    summary_stats = analyzer.calculate_summary_stats(per_query_scores)
    analyzer.print_summary(summary_stats)
    
    print(f"\n📊 PRESENTATION DATA READY:")
    print(f"  • CSV file: {csv_file} (for Excel/Google Sheets)")
    print(f"  • JSON file: {json_file} (for detailed analysis)")
    print(f"  • Total queries: {len(per_query_scores)}")
    
    return per_query_scores, summary_stats

if __name__ == "__main__":
    main()
