# Legal Chatbot Evaluation Metrics Guide

## Overview

This document describes the comprehensive evaluation metrics system integrated into the Philippine Law chatbot. The system measures both **Legal Information Accuracy** and **Content Relevance Rate** using standard classification metrics, automated scoring, and expert evaluation.

## Table of Contents

1. [Legal Information Accuracy Metrics](#legal-information-accuracy-metrics)
2. [Content Relevance Metrics](#content-relevance-metrics)
3. [Automated Content Scoring](#automated-content-scoring)
4. [Expert Evaluation Interface](#expert-evaluation-interface)
5. [Usage Examples](#usage-examples)
6. [Evaluation Tracking and Logging](#evaluation-tracking-and-logging)

---

## Legal Information Accuracy Metrics

### Composite Quality Score

The system uses a **composite quality score** (0.0 to 1.0) that combines multiple automated metrics to gauge legal information accuracy:

#### Components:

1. **Content Score (20%)**: BERTScore + ROUGE combination
   - Measures semantic similarity and lexical overlap
   - Range: 0.0 to 1.0

2. **Legal Elements (30%)**: Presence of key legal metadata
   - Checks completeness of case information
   - Range: 0.0 to 1.0

3. **Case Digest Accuracy (25%)**: Facts/Issues/Ruling identification
   - F1 score for section identification
   - Range: 0.0 to 1.0

4. **Completeness (10%)**: Response length and structure
   - Ensures adequate response detail
   - Range: 0.0 to 1.0

5. **Query Relevance (10%)**: Keyword and synonym matching
   - Measures alignment with query intent
   - Range: 0.0 to 1.0

### Usage Example

```python
from chatbot.evaluation_metrics import LegalAccuracyMetrics

# Assess legal information accuracy
accuracy_assessment = LegalAccuracyMetrics.assess_legal_information_accuracy(
    response="The Supreme Court ruled that...",
    reference_text="Reference legal text...",
    case_metadata={"case_title": "...", "gr_number": "..."},
    query="What was the ruling?"
)

print(f"Accuracy Score: {accuracy_assessment['accuracy_score']:.4f}")
print(f"Accuracy Level: {accuracy_assessment['accuracy_level']}")
print(f"Quality Threshold Met: {accuracy_assessment['quality_threshold_met']}")
print(f"Issues: {accuracy_assessment['accuracy_issues']}")
```

### Accuracy Levels

- **HIGH** (≥0.80): Excellent legal information accuracy
- **MEDIUM** (0.50-0.79): Moderate accuracy, improvements needed
- **LOW** (<0.50): Significant accuracy issues

---

## Content Relevance Metrics

### BLEU Score

**BLEU (Bilingual Evaluation Understudy)** measures the overlap between the chatbot response and reference legal texts.

- **BLEU-1**: Unigram overlap (individual words)
- **BLEU-2**: Bigram overlap (2-word phrases)
- **BLEU-3**: Trigram overlap (3-word phrases)
- **BLEU-4**: 4-gram overlap (4-word phrases)
- **BLEU-AVG**: Geometric mean of all BLEU scores

**Range**: 0.0 to 1.0 (higher indicates better overlap with reference)

**Use Case**: Ensures key elements from authoritative sources are present in the response.

### ROUGE Score

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** measures content overlap with focus on recall.

#### ROUGE-1
- Measures unigram overlap
- Provides precision, recall, and F1 score

#### ROUGE-2
- Measures bigram overlap
- Better for capturing phrasal accuracy

#### ROUGE-L
- Measures longest common subsequence
- Captures sentence-level structure similarity

**Range**: 0.0 to 1.0 for each metric (higher is better)

**Use Case**: Evaluates how much of the reference content is covered in the response.

### Example Usage

```python
from chatbot.evaluation_metrics import ContentRelevanceMetrics

candidate = "The Supreme Court held that due process was violated."
references = [
    "The Court ruled that procedural due process was infringed.",
    "The Supreme Court found a violation of due process rights."
]

# Calculate BLEU scores
bleu_scores = ContentRelevanceMetrics.calculate_bleu(candidate, references)
print(f"BLEU-1: {bleu_scores['bleu_1']}")
print(f"BLEU-2: {bleu_scores['bleu_2']}")
print(f"BLEU Average: {bleu_scores['bleu_avg']}")

# Calculate ROUGE scores
rouge_scores = ContentRelevanceMetrics.calculate_rouge(candidate, references)
print(f"ROUGE-1 F1: {rouge_scores['rouge_1']['f1']}")
print(f"ROUGE-2 F1: {rouge_scores['rouge_2']['f1']}")
print(f"ROUGE-L F1: {rouge_scores['rouge_l']['f1']}")
```

---

## Automated Content Scoring

The system provides comprehensive automated scoring that combines multiple metrics.

### Scoring Components

1. **BLEU Score** (25% weight)
   - Measures n-gram overlap with reference text

2. **ROUGE Score** (25% weight)
   - Average of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores

3. **Legal Elements Presence** (30% weight)
   - Checks for presence of key legal elements:
     - Case title
     - G.R. number
     - Ponente
     - Date
     - Case type
     - Facts
     - Issues
     - Ruling
     - Legal doctrine

4. **Citation Accuracy** (20% weight)
   - Precision, recall, and F1 of legal citations (G.R. numbers)

### Overall Relevance Score

**Formula**: 
```
Overall Score = (BLEU × 0.25) + (ROUGE × 0.25) + (Legal Elements × 0.30) + (Citations × 0.20)
```

**Range**: 0.0 to 1.0 (higher indicates better overall relevance)

### Example Usage

```python
from chatbot.evaluation_metrics import AutomatedContentScoring

response = """
**Case Title:** People v. Dela Cruz
**G.R. No.:** 123456
**Ponente:** Justice Santos
**Date:** January 15, 2020

**Facts:** The petitioner was charged with theft...
**Issues:** Whether guilt was proven beyond reasonable doubt...
**Ruling:** The Supreme Court ruled in favor of the accused...
"""

reference = "Full case text from JSONL..."

case_metadata = {
    'case_title': 'People v. Dela Cruz',
    'gr_number': 'G.R. No. 123456',
    'ponente': 'Justice Santos',
    'promulgation_date': '2020-01-15',
    'case_type': 'Criminal'
}

scores = AutomatedContentScoring.score_legal_response(
    response, reference, case_metadata
)

print(f"Overall Relevance: {scores['overall_relevance']['score']}")
print(f"BLEU: {scores['bleu']['bleu_avg']}")
print(f"ROUGE-1 F1: {scores['rouge']['rouge_1']['f1']}")
print(f"Legal Elements: {scores['legal_elements']['presence_rate']}")
print(f"Citation Accuracy: {scores['citation_accuracy']['f1']}")
```

---

## Expert Evaluation Interface

The system includes an interface for human expert evaluation.

### Evaluation Criteria

Experts rate responses on a 1-5 scale for:

1. **Accuracy**: Factual correctness of legal information
2. **Completeness**: Coverage of all relevant legal aspects
3. **Relevance**: Relevance to the user query
4. **Clarity**: Clarity and understandability
5. **Legal Reasoning**: Quality of legal reasoning and analysis
6. **Citation Accuracy**: Accuracy of legal citations

### Example Usage

```python
from chatbot.expert_evaluation import ExpertEvaluationInterface

# Create expert interface
expert = ExpertEvaluationInterface(expert_id="legal_expert_001")

# Create evaluation form
form = expert.create_evaluation_form(
    query="What is the ruling in G.R. No. 123456?",
    response="The Supreme Court ruled...",
    case_id="GR-123456"
)

# Expert provides scores
form['scores'] = {
    'accuracy': 4,
    'completeness': 3,
    'relevance': 5,
    'clarity': 4,
    'legal_reasoning': 4,
    'citation_accuracy': 3
}
form['comments'] = "Good response, but needs more detail on reasoning."

# Submit evaluation
expert.submit_evaluation(form)

# Get statistics
stats = expert.get_expert_statistics()
print(f"Total Evaluations: {stats['total_evaluations']}")
print(f"Average Rating: {stats['average_overall_rating']}/5")
```

### Batch Evaluation

For evaluating multiple responses:

```python
from chatbot.expert_evaluation import create_batch_evaluation_set

queries_and_responses = [
    {
        'query': 'What is the ruling in G.R. No. 123456?',
        'response': 'The Supreme Court ruled...',
        'case_id': 'GR-123456'
    },
    # ... more items
]

batch_file = create_batch_evaluation_set(
    queries_and_responses, 
    expert_id="legal_expert_001"
)

# Expert fills out the batch file, then:
from chatbot.expert_evaluation import load_and_submit_batch_evaluations

results = load_and_submit_batch_evaluations(batch_file)
print(f"Submitted: {results['submitted']}/{results['total']}")
```

---

## Usage Examples

### Automatic Evaluation in Chat Engine

The chat engine automatically evaluates responses when generating case digests:

```python
from chatbot.chat_engine import chat_with_law_bot, get_evaluation_statistics

# Query the chatbot
response = chat_with_law_bot("What is the ruling in G.R. No. 123456?")

# Evaluation is automatically logged
# Get session statistics
stats = get_evaluation_statistics()

print(f"Session Evaluations: {stats['total_evaluations']}")
print(f"Average BLEU: {stats['average_scores']['bleu_avg']}")
print(f"Average ROUGE-1 F1: {stats['average_scores']['rouge_1_f1']}")
print(f"Average Overall Relevance: {stats['average_scores']['overall_relevance']}")
```

### Manual Evaluation

To manually evaluate a response:

```python
from chatbot.chat_engine import evaluate_response

query = "What is the ruling in G.R. No. 123456?"
response = "The Supreme Court held..."
reference_text = "Full case text from JSONL..."

case_metadata = {
    'case_title': 'People v. Dela Cruz',
    'gr_number': 'G.R. No. 123456',
    'ponente': 'Justice Santos',
    'promulgation_date': '2020-01-15',
    'case_type': 'Criminal'
}

evaluation = evaluate_response(
    query, 
    response, 
    reference_text, 
    case_metadata
)

# Evaluation results are printed and logged
```

---

## Evaluation Tracking and Logging

### Automatic Logging

All evaluations are automatically logged to JSONL files:

- **Location**: `backend/data/evaluation_logs/`
- **Format**: `evaluation_YYYYMMDD_HHMMSS.jsonl`
- **Content**: Each line contains one evaluation record with all metrics

### Log Structure

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "session_id": "20250115_103000",
  "query": "What is the ruling in G.R. No. 123456?",
  "response": "The Supreme Court held...",
  "reference": "Full case text (first 500 chars)...",
  "case_metadata": {...},
  "automated_scores": {
    "bleu": {...},
    "rouge": {...},
    "legal_elements": {...},
    "citation_accuracy": {...},
    "overall_relevance": {...}
  },
  "expert_scores": null,
  "response_length": 1234,
  "reference_length": 5678
}
```

### Session Statistics

Get aggregated statistics for the current session:

```python
from chatbot.chat_engine import get_evaluation_statistics

stats = get_evaluation_statistics()

print(f"Total Evaluations: {stats['total_evaluations']}")
print(f"\nAverage Scores:")
for metric, value in stats['average_scores'].items():
    print(f"  {metric}: {value:.4f}")

print(f"\nStandard Deviations:")
for metric, value in stats['std_scores'].items():
    print(f"  {metric}: {value:.4f}")
```

### Export Report

Export evaluation report to JSON:

```python
from chatbot.evaluation_metrics import EvaluationTracker

tracker = EvaluationTracker()
report_file = tracker.export_report()
print(f"Report saved to: {report_file}")
```

---

## Testing

Run the test suite to verify metrics are working:

```bash
cd backend
python test_evaluation_metrics.py
```

Expected output includes:
- Binary and multiclass classification metrics
- BLEU and ROUGE scores
- Automated content scoring results
- Evaluation tracking statistics

---

## Best Practices

1. **Regular Monitoring**: Review evaluation logs regularly to identify areas for improvement

2. **Expert Validation**: Combine automated metrics with expert evaluation for comprehensive assessment

3. **Metric Interpretation**:
   - BLEU > 0.3: Good overlap with reference
   - ROUGE-L F1 > 0.5: Strong content coverage
   - Legal Elements > 0.7: Most key elements present
   - Overall Relevance > 0.6: High-quality response

4. **Continuous Improvement**: Use metrics to guide model fine-tuning and prompt engineering

5. **Batch Evaluation**: Periodically conduct batch expert evaluations on random samples

---

## File Structure

```
backend/
├── chatbot/
│   ├── evaluation_metrics.py      # Core metrics implementation
│   ├── expert_evaluation.py       # Expert evaluation interface
│   ├── chat_engine.py             # Integration with chat engine
│   └── ...
├── data/
│   ├── evaluation_logs/           # Automatic evaluation logs
│   └── expert_evaluations/        # Expert evaluation data
├── test_evaluation_metrics.py     # Test suite
└── EVALUATION_METRICS_GUIDE.md    # This file
```

---

## References

- **BLEU**: Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation"
- **ROUGE**: Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries"
- **Classification Metrics**: Standard machine learning evaluation metrics

---

## Support

For questions or issues with the evaluation metrics system, contact the development team or refer to the test suite for usage examples.
