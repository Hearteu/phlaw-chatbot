# Rating Metrics Guide

This guide explains how to use the rating system to calculate legal information accuracy metrics like F1 score, precision, recall, etc.

## Overview

The rating system collects user feedback on chatbot responses and uses this data to calculate comprehensive performance metrics:

- **Legal Information Accuracy**: Classification metrics (accuracy, precision, recall, F1 score, specificity)
- **Content Quality**: User satisfaction metrics (helpfulness, clarity, confidence)

## How It Works

### 1. Data Collection
Users rate responses through the frontend interface:
- **Correctness**: Yes/No (used for accuracy metrics)
- **Confidence**: 1-5 stars (user's confidence in the response)
- **Helpfulness**: 1-5 stars (how helpful the response was)
- **Clarity**: 1-5 stars (how clear the response was)
- **Comments**: Optional feedback text

### 2. Data Storage
Ratings are stored in JSONL files:
```
backend/data/ratings/ratings_YYYYMMDD.jsonl
```

Each rating contains:
```json
{
  "query": "G.R. No. 158563",
  "response": "Response text...",
  "case_id": "G.R. No. 158563",
  "correctness": true,
  "confidence": 4,
  "helpfulness": 4,
  "clarity": 5,
  "comment": "Good response",
  "user_id": "student1",
  "timestamp": 1735689600
}
```

### 3. Metrics Calculation

#### Legal Accuracy Metrics
- **Accuracy**: (True Positives + True Negatives) / Total
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Specificity**: True Negatives / (True Negatives + False Positives)

#### Content Quality Metrics
- **Average Helpfulness**: Mean of all helpfulness ratings (1-5)
- **Average Clarity**: Mean of all clarity ratings (1-5)
- **Average Confidence**: Mean of all confidence ratings (1-5)

## Usage

### 1. Command Line Analysis

```bash
# Analyze ratings for last 30 days
python manage.py analyze_ratings

# Analyze ratings for last 7 days with detailed output
python manage.py analyze_ratings --days 7 --verbose

# Export results to JSON file
python manage.py analyze_ratings --export --output my_analysis.json
```

### 2. API Endpoints

#### Get Rating Metrics
```http
GET /api/rating/metrics/?days=30&details=true
```

Response:
```json
{
  "period_days": 30,
  "total_ratings": 150,
  "overall_metrics": {
    "accuracy": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.88,
      "f1_score": 0.85,
      "specificity": 0.83,
      "total_ratings": 150
    },
    "content": {
      "avg_helpfulness": 4.2,
      "avg_clarity": 4.0,
      "avg_confidence": 3.8,
      "total_ratings": 150
    }
  },
  "summary": "LEGAL CHATBOT PERFORMANCE SUMMARY...",
  "user_analysis": {...},
  "case_analysis": {...}
}
```

### 3. Programmatic Usage

```python
from chatbot.rating_analyzer import RatingAnalyzer

# Initialize analyzer
analyzer = RatingAnalyzer()

# Generate comprehensive report
report = analyzer.generate_report(days_back=30)

# Get specific metrics
accuracy_metrics = analyzer.calculate_accuracy_metrics(ratings)
content_metrics = analyzer.calculate_content_metrics(ratings)

# Analyze by user
user_analysis = analyzer.analyze_by_user(ratings)

# Analyze by case type
case_analysis = analyzer.analyze_by_case_type(ratings)
```

## Frontend Integration

### Metrics Display Component
Use the `MetricsDisplay` component to show metrics in the frontend:

```tsx
import MetricsDisplay from '@/components/MetricsDisplay';

function AdminDashboard() {
  return (
    <div>
      <h1>Chatbot Performance</h1>
      <MetricsDisplay />
    </div>
  );
}
```

The component provides:
- Interactive time period selection (7, 30, 90 days)
- Visual progress bars for all metrics
- Summary statistics cards
- Detailed breakdown by user and case type

## Interpreting Results

### Accuracy Metrics
- **Accuracy > 80%**: Good performance
- **F1 Score > 0.8**: Well-balanced precision and recall
- **Precision > 0.8**: Low false positive rate
- **Recall > 0.8**: Low false negative rate

### Content Quality
- **Helpfulness > 4.0/5**: Users find responses helpful
- **Clarity > 4.0/5**: Responses are clear and understandable
- **Confidence > 3.5/5**: Users are confident in responses

### Sample Interpretation
```
LEGAL CHATBOT PERFORMANCE SUMMARY
================================
Total Ratings: 150

LEGAL ACCURACY METRICS:
- Overall Accuracy: 85.0%
- F1 Score: 85.0%
- Precision: 82.0%
- Recall: 88.0%

CONTENT QUALITY METRICS:
- Average Helpfulness: 4.2/5.0
- Average Clarity: 4.0/5.0
- Average Confidence: 3.8/5.0

INTERPRETATION:
- Accuracy 85.0% means the chatbot was rated as correct in 85.0% of cases
- F1 Score 85.0% indicates good balance between precision and recall
- Helpfulness 4.2/5 suggests excellent user satisfaction
```

## Advanced Analysis

### User Segmentation
Analyze performance by user type:
- Students vs. Experts
- Frequent vs. Occasional users
- Different user groups

### Case Type Analysis
Break down performance by legal area:
- Criminal law
- Civil law
- Administrative law
- Constitutional law

### Temporal Analysis
Track performance over time:
- Daily/weekly trends
- Performance improvements
- Seasonal variations

## Best Practices

1. **Collect Sufficient Data**: Aim for at least 100 ratings for reliable metrics
2. **Regular Analysis**: Run analysis weekly to track trends
3. **User Feedback**: Use comments to identify specific improvement areas
4. **A/B Testing**: Compare metrics before/after system changes
5. **Expert Validation**: Periodically validate user ratings with expert review

## Troubleshooting

### No Ratings Found
- Check if ratings directory exists: `backend/data/ratings/`
- Verify rating files are being created
- Check date range in analysis

### Low Accuracy Scores
- Review user comments for common issues
- Check if responses are actually incorrect
- Consider if rating criteria are clear

### Inconsistent Metrics
- Ensure sufficient sample size
- Check for data quality issues
- Verify rating collection process

## Files and Components

### Backend
- `chatbot/rating_analyzer.py` - Main analysis logic
- `chatbot/management/commands/analyze_ratings.py` - Django command
- `chatbot/views.py` - API endpoints
- `chatbot/evaluation_metrics.py` - Metrics calculation

### Frontend
- `components/MetricsDisplay.tsx` - Metrics visualization
- `components/RatingComponent.tsx` - Rating interface

### Data
- `data/ratings/ratings_*.jsonl` - Rating data files
- `data/analysis/` - Exported analysis reports
