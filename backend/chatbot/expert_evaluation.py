# expert_evaluation.py — Expert evaluation interface for human scoring
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class ExpertEvaluationInterface:
    """Interface for expert evaluation of chatbot responses"""
    
    EVALUATION_CRITERIA = {
        'accuracy': {
            'description': 'Factual correctness of legal information',
            'scale': '1-5 (1=Incorrect, 5=Completely Accurate)'
        },
        'completeness': {
            'description': 'Coverage of all relevant legal aspects',
            'scale': '1-5 (1=Incomplete, 5=Comprehensive)'
        },
        'relevance': {
            'description': 'Relevance to the user query',
            'scale': '1-5 (1=Not Relevant, 5=Highly Relevant)'
        },
        'clarity': {
            'description': 'Clarity and understandability of response',
            'scale': '1-5 (1=Confusing, 5=Very Clear)'
        },
        'legal_reasoning': {
            'description': 'Quality of legal reasoning and analysis',
            'scale': '1-5 (1=Poor, 5=Excellent)'
        },
        'citation_accuracy': {
            'description': 'Accuracy of legal citations and references',
            'scale': '1-5 (1=Incorrect, 5=Accurate)'
        }
    }
    
    def __init__(self, expert_id: str, save_dir: str = "backend/data/expert_evaluations"):
        self.expert_id = expert_id
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.evaluation_file = os.path.join(save_dir, f"expert_{expert_id}_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    def create_evaluation_form(self, query: str, response: str, case_id: Optional[str] = None) -> Dict:
        """Create an evaluation form for expert review"""
        form = {
            'evaluation_id': f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'expert_id': self.expert_id,
            'timestamp': datetime.now().isoformat(),
            'case_id': case_id,
            'query': query,
            'response': response,
            'criteria': self.EVALUATION_CRITERIA,
            'scores': {},
            'comments': "",
            'overall_rating': 0
        }
        return form
    
    def submit_evaluation(self, evaluation_form: Dict) -> bool:
        """Submit completed evaluation"""
        # Validate scores
        for criterion in self.EVALUATION_CRITERIA.keys():
            if criterion not in evaluation_form['scores']:
                print(f"Warning: Missing score for {criterion}")
                return False
            
            score = evaluation_form['scores'][criterion]
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                print(f"Error: Invalid score for {criterion}: {score}")
                return False
        
        # Calculate overall rating (average of all criteria)
        overall = sum(evaluation_form['scores'].values()) / len(evaluation_form['scores'])
        evaluation_form['overall_rating'] = round(overall, 2)
        
        # Save to file
        try:
            with open(self.evaluation_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(evaluation_form, ensure_ascii=False) + '\n')
            print(f"Evaluation saved: {evaluation_form['evaluation_id']}")
            return True
        except Exception as e:
            print(f"Error saving evaluation: {e}")
            return False
    
    def get_expert_statistics(self) -> Dict:
        """Get statistics for this expert's evaluations"""
        if not os.path.exists(self.evaluation_file):
            return {}
        
        evaluations = []
        with open(self.evaluation_file, 'r', encoding='utf-8') as f:
            for line in f:
                evaluations.append(json.loads(line))
        
        if not evaluations:
            return {}
        
        # Calculate average scores per criterion
        criteria_averages = {}
        for criterion in self.EVALUATION_CRITERIA.keys():
            scores = [e['scores'][criterion] for e in evaluations if criterion in e['scores']]
            if scores:
                criteria_averages[criterion] = round(sum(scores) / len(scores), 2)
        
        # Overall statistics
        overall_ratings = [e['overall_rating'] for e in evaluations]
        
        return {
            'expert_id': self.expert_id,
            'total_evaluations': len(evaluations),
            'average_overall_rating': round(sum(overall_ratings) / len(overall_ratings), 2),
            'criteria_averages': criteria_averages,
            'evaluation_file': self.evaluation_file
        }
    
    @staticmethod
    def print_evaluation_form_template():
        """Print template for manual evaluation"""
        print("\n" + "="*70)
        print("EXPERT EVALUATION FORM TEMPLATE")
        print("="*70)
        print("\nEvaluation Criteria (Score each 1-5):\n")
        
        for criterion, details in ExpertEvaluationInterface.EVALUATION_CRITERIA.items():
            print(f"{criterion.upper().replace('_', ' ')}:")
            print(f"  Description: {details['description']}")
            print(f"  Scale: {details['scale']}")
            print()
        
        print("OVERALL COMMENTS:")
        print("  (Provide detailed feedback on the response)")
        print("\n" + "="*70)


def create_batch_evaluation_set(queries_and_responses: List[Dict], expert_id: str) -> str:
    """
    Create a batch evaluation set for expert review
    
    Args:
        queries_and_responses: List of {'query': str, 'response': str, 'case_id': str}
        expert_id: Expert identifier
        
    Returns:
        Path to evaluation batch file
    """
    interface = ExpertEvaluationInterface(expert_id)
    
    batch_file = os.path.join(
        interface.save_dir, 
        f"batch_{expert_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    batch_data = {
        'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'expert_id': expert_id,
        'created_at': datetime.now().isoformat(),
        'total_items': len(queries_and_responses),
        'items': []
    }
    
    for idx, item in enumerate(queries_and_responses):
        form = interface.create_evaluation_form(
            query=item['query'],
            response=item['response'],
            case_id=item.get('case_id')
        )
        form['item_number'] = idx + 1
        batch_data['items'].append(form)
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)
    
    print(f"Batch evaluation set created: {batch_file}")
    print(f"Total items: {len(queries_and_responses)}")
    
    return batch_file


def load_and_submit_batch_evaluations(batch_file: str) -> Dict:
    """
    Load completed batch evaluations and submit them
    
    Args:
        batch_file: Path to completed batch evaluation file
        
    Returns:
        Summary of submission results
    """
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    expert_id = batch_data['expert_id']
    interface = ExpertEvaluationInterface(expert_id)
    
    results = {
        'total': len(batch_data['items']),
        'submitted': 0,
        'failed': 0,
        'errors': []
    }
    
    for item in batch_data['items']:
        if 'scores' not in item or not item['scores']:
            print(f"Skipping item {item.get('item_number')} - no scores provided")
            results['failed'] += 1
            continue
        
        if interface.submit_evaluation(item):
            results['submitted'] += 1
        else:
            results['failed'] += 1
            results['errors'].append(f"Item {item.get('item_number')} submission failed")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("EXPERT EVALUATION INTERFACE - EXAMPLE USAGE")
    print("="*70)
    
    # Print evaluation template
    ExpertEvaluationInterface.print_evaluation_form_template()
    
    # Create example evaluation
    expert = ExpertEvaluationInterface(expert_id="legal_expert_001")
    
    form = expert.create_evaluation_form(
        query="What is the ruling in G.R. No. 123456?",
        response="The Supreme Court ruled in favor of the petitioner...",
        case_id="GR-123456"
    )
    
    # Example scores (would be provided by expert)
    form['scores'] = {
        'accuracy': 4,
        'completeness': 3,
        'relevance': 5,
        'clarity': 4,
        'legal_reasoning': 4,
        'citation_accuracy': 3
    }
    form['comments'] = "Good overall response, but could include more details on the legal reasoning."
    
    # Submit evaluation
    if expert.submit_evaluation(form):
        print("\nExample evaluation submitted successfully!")
        
        # Get statistics
        stats = expert.get_expert_statistics()
        print("\nExpert Statistics:")
        print(f"  Total Evaluations: {stats['total_evaluations']}")
        print(f"  Average Overall Rating: {stats['average_overall_rating']}/5")
        print("\n  Criteria Averages:")
        for criterion, avg in stats['criteria_averages'].items():
            print(f"    {criterion}: {avg}/5")
