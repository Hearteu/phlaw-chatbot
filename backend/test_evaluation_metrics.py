# test_evaluation_metrics.py — Test evaluation metrics integration
import sys

sys.path.insert(0, '.')

from chatbot.evaluation_metrics import (AutomatedContentScoring,
                                        ContentRelevanceMetrics,
                                        EvaluationTracker,
                                        LegalAccuracyMetrics)


def test_accuracy_metrics():
    """Test legal information accuracy metrics"""
    print("=" * 60)
    print("Testing Legal Information Accuracy Metrics")
    print("=" * 60)
    
    # Test binary classification
    predictions = [1, 1, 0, 1, 0, 0, 1, 0]
    ground_truth = [1, 0, 0, 1, 0, 1, 1, 0]
    
    metrics = LegalAccuracyMetrics.calculate_metrics(predictions, ground_truth)
    
    print("\nBinary Classification Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}")
    print(f"  FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    
    # Test multiclass classification
    predictions_multi = [0, 1, 2, 0, 1, 2, 0, 1]
    ground_truth_multi = [0, 1, 1, 0, 2, 2, 0, 1]
    
    metrics_multi = LegalAccuracyMetrics.calculate_multiclass_metrics(
        predictions_multi, ground_truth_multi, num_classes=3
    )
    
    print("\n\nMulticlass Classification Metrics:")
    print(f"  Macro Precision: {metrics_multi['macro_avg']['precision']:.4f}")
    print(f"  Macro Recall:    {metrics_multi['macro_avg']['recall']:.4f}")
    print(f"  Macro F1 Score:  {metrics_multi['macro_avg']['f1_score']:.4f}")


def test_content_relevance_metrics():
    """Test BLEU and ROUGE metrics"""
    print("\n" + "=" * 60)
    print("Testing Content Relevance Metrics (BLEU & ROUGE)")
    print("=" * 60)
    
    candidate = """
    The Supreme Court held that the petitioner's right to due process was violated. 
    The court ruled in favor of the respondent and ordered the payment of damages.
    The doctrine of res judicata applies in this case.
    """
    
    references = [
        """
        The Supreme Court ruled that due process rights were infringed upon.
        The petitioner's claim was denied and damages were awarded to the respondent.
        Res judicata prevents relitigation of the same matter.
        """,
        """
        In its decision, the Supreme Court found a violation of procedural due process.
        The ruling favored the respondent with an award of compensatory damages.
        The principle of res judicata bars this action.
        """
    ]
    
    # Calculate BLEU scores
    bleu_scores = ContentRelevanceMetrics.calculate_bleu(candidate, references)
    print("\nBLEU Scores:")
    for key, value in bleu_scores.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    # Calculate ROUGE scores
    rouge_scores = ContentRelevanceMetrics.calculate_rouge(candidate, references)
    print("\nROUGE Scores:")
    for rouge_type, scores in rouge_scores.items():
        print(f"\n  {rouge_type.upper()}:")
        print(f"    Precision: {scores['precision']:.4f}")
        print(f"    Recall:    {scores['recall']:.4f}")
        print(f"    F1 Score:  {scores['f1']:.4f}")


def test_automated_scoring():
    """Test automated content scoring"""
    print("\n" + "=" * 60)
    print("Testing Automated Content Scoring")
    print("=" * 60)
    
    response = """
    **Case Title:** People of the Philippines v. Juan Dela Cruz
    **G.R. No.:** 123456
    **Ponente:** Justice Santos
    **Date:** January 15, 2020
    **Case Type:** Criminal
    
    **Facts:**
    The petitioner was charged with theft. The prosecution alleged that the petitioner 
    took property belonging to the respondent without consent.
    
    **Issues:**
    Whether the petitioner is guilty of theft beyond reasonable doubt.
    
    **Ruling:**
    The Supreme Court held that the prosecution failed to prove guilt beyond reasonable doubt.
    The doctrine of reasonable doubt applies. The petition is GRANTED.
    """
    
    reference = """
    People of the Philippines v. Juan Dela Cruz, G.R. No. 123456
    
    This is a criminal case decided by Justice Santos on January 15, 2020.
    
    The facts show that the accused was charged with theft of property.
    The prosecution presented evidence that the accused took items without permission.
    
    The legal issue is whether guilt was proven beyond reasonable doubt.
    
    The Supreme Court ruled in favor of the accused, finding insufficient evidence.
    The burden of proof was not met by the prosecution.
    The principle of reasonable doubt requires acquittal when evidence is lacking.
    
    WHEREFORE, the petition is GRANTED and the accused is acquitted.
    """
    
    case_metadata = {
        'case_title': 'People of the Philippines v. Juan Dela Cruz',
        'gr_number': 'G.R. No. 123456',
        'ponente': 'Justice Santos',
        'promulgation_date': '2020-01-15',
        'case_type': 'Criminal'
    }
    
    scores = AutomatedContentScoring.score_legal_response(response, reference, case_metadata)
    
    print("\nAutomated Content Scoring Results:")
    print(f"\n  BLEU Average: {scores['bleu']['bleu_avg']:.4f}")
    print(f"  ROUGE-1 F1:   {scores['rouge']['rouge_1']['f1']:.4f}")
    print(f"  ROUGE-2 F1:   {scores['rouge']['rouge_2']['f1']:.4f}")
    print(f"  ROUGE-L F1:   {scores['rouge']['rouge_l']['f1']:.4f}")
    
    print(f"\n  Legal Elements Presence Rate: {scores['legal_elements']['presence_rate']:.4f}")
    print(f"  Elements Found: {scores['legal_elements']['elements_found']}/{scores['legal_elements']['total_elements']}")
    
    print(f"\n  Citation Accuracy F1: {scores['citation_accuracy']['f1']:.4f}")
    
    print(f"\n  Overall Relevance Score: {scores['overall_relevance']['score']:.4f}")
    print(f"    BLEU Component:          {scores['overall_relevance']['bleu_component']:.4f}")
    print(f"    ROUGE Component:         {scores['overall_relevance']['rouge_component']:.4f}")
    print(f"    Legal Elements Component: {scores['overall_relevance']['legal_elements_component']:.4f}")
    print(f"    Citation Component:      {scores['overall_relevance']['citation_component']:.4f}")


def test_evaluation_tracker():
    """Test evaluation tracking and logging"""
    print("\n" + "=" * 60)
    print("Testing Evaluation Tracker")
    print("=" * 60)
    
    tracker = EvaluationTracker()
    
    # Log a few evaluations
    for i in range(3):
        query = f"What is the ruling in G.R. No. {123456 + i}?"
        response = f"The Supreme Court ruled in favor of the petitioner in case {i+1}."
        reference = f"In G.R. No. {123456 + i}, the Court granted the petition."
        
        tracker.log_evaluation(query, response, reference)
    
    print(f"\n  Session ID: {tracker.session_id}")
    print(f"  Log File: {tracker.log_file}")
    
    # Get statistics
    stats = tracker.get_session_statistics()
    if stats:
        print(f"\n  Total Evaluations: {stats['total_evaluations']}")
        print(f"\n  Average Scores:")
        for metric, value in stats['average_scores'].items():
            print(f"    {metric}: {value:.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LEGAL CHATBOT EVALUATION METRICS TEST SUITE")
    print("="*60)
    
    try:
        test_accuracy_metrics()
        test_content_relevance_metrics()
        test_automated_scoring()
        test_evaluation_tracker()
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
