#!/usr/bin/env python3
"""
Rating Analyzer - Process user ratings and calculate legal accuracy metrics
"""
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .evaluation_metrics import ContentRelevanceMetrics, LegalAccuracyMetrics


class RatingAnalyzer:
    """Analyze user ratings and calculate legal accuracy metrics"""
    
    def __init__(self, ratings_dir: str = "data/ratings"):
        self.ratings_dir = ratings_dir
        self.legal_metrics = LegalAccuracyMetrics()
        self.content_metrics = ContentRelevanceMetrics()
    
    def load_ratings(self, days_back: int = 30) -> List[Dict]:
        """Load ratings from JSONL files"""
        ratings = []
        
        # Get all rating files
        pattern = os.path.join(self.ratings_dir, "ratings_*.jsonl")
        rating_files = glob.glob(pattern)
        
        if not rating_files:
            print(f"No rating files found in {self.ratings_dir}")
            return ratings
        
        # Load ratings from the last N days
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for file_path in sorted(rating_files):
            try:
                # Extract date from filename
                filename = os.path.basename(file_path)
                date_str = filename.replace("ratings_", "").replace(".jsonl", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            rating = json.loads(line.strip())
                            ratings.append(rating)
                            
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(ratings)} ratings from {len(rating_files)} files")
        return ratings
    
    def calculate_accuracy_metrics(self, ratings: List[Dict]) -> Dict[str, float]:
        """Calculate legal accuracy metrics from ratings"""
        if not ratings:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
                'total_ratings': 0
            }
        
        # Extract correctness ratings (ground truth)
        ground_truth = [1 if rating.get('correctness', False) else 0 for rating in ratings]
        
        # For now, we'll use the correctness as both prediction and ground truth
        # In a real scenario, you might have expert labels as ground truth
        predictions = ground_truth.copy()
        
        # Calculate metrics
        metrics = self.legal_metrics.calculate_metrics(predictions, ground_truth)
        metrics['total_ratings'] = len(ratings)
        
        return metrics
    
    def calculate_content_metrics(self, ratings: List[Dict]) -> Dict[str, float]:
        """Calculate content relevance metrics from ratings"""
        if not ratings:
            return {
                'avg_helpfulness': 0.0,
                'avg_clarity': 0.0,
                'avg_confidence': 0.0,
                'total_ratings': 0
            }
        
        # Extract star ratings
        helpfulness_scores = [rating.get('helpfulness', 0) for rating in ratings]
        clarity_scores = [rating.get('clarity', 0) for rating in ratings]
        confidence_scores = [rating.get('confidence', 0) for rating in ratings]
        
        # Calculate averages
        metrics = {
            'avg_helpfulness': sum(helpfulness_scores) / len(helpfulness_scores) if helpfulness_scores else 0.0,
            'avg_clarity': sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0,
            'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            'total_ratings': len(ratings)
        }
        
        return metrics
    
    def analyze_by_user(self, ratings: List[Dict]) -> Dict[str, Dict]:
        """Analyze ratings grouped by user"""
        user_ratings = defaultdict(list)
        
        for rating in ratings:
            user_id = rating.get('user_id', 'anonymous')
            user_ratings[user_id].append(rating)
        
        user_analysis = {}
        for user_id, user_ratings_list in user_ratings.items():
            user_analysis[user_id] = {
                'accuracy_metrics': self.calculate_accuracy_metrics(user_ratings_list),
                'content_metrics': self.calculate_content_metrics(user_ratings_list),
                'rating_count': len(user_ratings_list)
            }
        
        return user_analysis
    
    def analyze_by_case_type(self, ratings: List[Dict]) -> Dict[str, Dict]:
        """Analyze ratings grouped by case type (if available)"""
        case_ratings = defaultdict(list)
        
        for rating in ratings:
            # Try to extract case type from response content
            response = rating.get('response', '')
            case_type = self._extract_case_type(response)
            case_ratings[case_type].append(rating)
        
        case_analysis = {}
        for case_type, case_ratings_list in case_ratings.items():
            case_analysis[case_type] = {
                'accuracy_metrics': self.calculate_accuracy_metrics(case_ratings_list),
                'content_metrics': self.calculate_content_metrics(case_ratings_list),
                'rating_count': len(case_ratings_list)
            }
        
        return case_analysis
    
    def _extract_case_type(self, response: str) -> str:
        """Extract case type from response content"""
        response_lower = response.lower()
        
        if 'criminal' in response_lower:
            return 'criminal'
        elif 'civil' in response_lower:
            return 'civil'
        elif 'administrative' in response_lower:
            return 'administrative'
        elif 'constitutional' in response_lower:
            return 'constitutional'
        elif 'labor' in response_lower:
            return 'labor'
        elif 'family' in response_lower:
            return 'family'
        else:
            return 'other'
    
    def generate_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        print(f"Generating rating analysis report for last {days_back} days...")
        
        # Load ratings
        ratings = self.load_ratings(days_back)
        
        if not ratings:
            return {
                'error': 'No ratings found',
                'period_days': days_back,
                'total_ratings': 0
            }
        
        # Calculate overall metrics
        accuracy_metrics = self.calculate_accuracy_metrics(ratings)
        content_metrics = self.calculate_content_metrics(ratings)
        
        # Calculate detailed analysis
        user_analysis = self.analyze_by_user(ratings)
        case_analysis = self.analyze_by_case_type(ratings)
        
        # Generate report
        report = {
            'period_days': days_back,
            'total_ratings': len(ratings),
            'overall_metrics': {
                'accuracy': accuracy_metrics,
                'content': content_metrics
            },
            'user_analysis': user_analysis,
            'case_analysis': case_analysis,
            'summary': self._generate_summary(accuracy_metrics, content_metrics, len(ratings))
        }
        
        return report
    
    def _generate_summary(self, accuracy_metrics: Dict, content_metrics: Dict, total_ratings: int) -> str:
        """Generate human-readable summary"""
        accuracy = accuracy_metrics.get('accuracy', 0) * 100
        f1_score = accuracy_metrics.get('f1_score', 0) * 100
        avg_helpfulness = content_metrics.get('avg_helpfulness', 0)
        avg_clarity = content_metrics.get('avg_clarity', 0)
        
        summary = f"""
LEGAL CHATBOT PERFORMANCE SUMMARY
================================
Total Ratings: {total_ratings}

LEGAL ACCURACY METRICS:
- Overall Accuracy: {accuracy:.1f}%
- F1 Score: {f1_score:.1f}%
- Precision: {accuracy_metrics.get('precision', 0)*100:.1f}%
- Recall: {accuracy_metrics.get('recall', 0)*100:.1f}%

CONTENT QUALITY METRICS:
- Average Helpfulness: {avg_helpfulness:.1f}/5.0
- Average Clarity: {avg_clarity:.1f}/5.0
- Average Confidence: {content_metrics.get('avg_confidence', 0):.1f}/5.0

INTERPRETATION:
- Accuracy {accuracy:.1f}% means the chatbot was rated as correct in {accuracy:.1f}% of cases
- F1 Score {f1_score:.1f}% indicates the balance between precision and recall
- Helpfulness {avg_helpfulness:.1f}/5 suggests {'excellent' if avg_helpfulness >= 4 else 'good' if avg_helpfulness >= 3 else 'needs improvement'} user satisfaction
        """
        
        return summary.strip()
    
    def export_metrics(self, report: Dict, output_file: str = None) -> str:
        """Export metrics to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rating_analysis_{timestamp}.json"
        
        output_path = os.path.join(self.ratings_dir, "..", "analysis", output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis exported to: {output_path}")
        return output_path


def main():
    """Command-line interface for rating analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze chatbot ratings and calculate metrics')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--export', action='store_true', help='Export results to JSON file')
    parser.add_argument('--output', type=str, help='Output file name for export')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RatingAnalyzer()
    
    # Generate report
    report = analyzer.generate_report(days_back=args.days)
    
    # Print summary
    if 'summary' in report:
        print(report['summary'])
    else:
        print(f"Error: {report.get('error', 'Unknown error')}")
        return 1
    
    # Export if requested
    if args.export:
        output_file = analyzer.export_metrics(report, args.output)
        print(f"Results exported to: {output_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
