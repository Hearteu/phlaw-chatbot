# PHLaw Chatbot Testing Suite

This directory contains scripts for testing and analyzing the PHLaw chatbot's performance.

## Scripts

### 1. `test_chatbot_queries.py`
Runs a series of test queries against the chatbot and saves results to JSON.

### 2. `analyze_test_results.py`
Analyzes test results and generates detailed performance reports.

## Quick Start

### Running Tests

```bash
# From the backend directory
cd backend

# Run the test suite
python test_chatbot_queries.py
```

This will:
- Execute 5 different test queries covering various scenarios
- Log query and response details
- Save results to `test_results/chatbot_test_YYYYMMDD_HHMMSS.json`
- Display a summary in the console

### Analyzing Results

```bash
# Analyze the latest test run
python analyze_test_results.py latest

# Or analyze a specific test file
python analyze_test_results.py test_results/chatbot_test_20250117_143022.json
```

This will:
- Generate detailed quality metrics
- Analyze performance by query type
- Show individual test results
- Save analysis to `*_analysis.txt`

## Test Query Types

The test suite covers 5 different query scenarios:

1. **GR Number Lookup**: `"What is the ruling in G.R. No. 161295?"`
   - Tests direct case retrieval by ID

2. **Legal Concept**: `"What are the legal principles regarding copyright infringement?"`
   - Tests semantic search and synthesis

3. **Legal Question**: `"Can a lawyer represent a client in a case related to their previous government work?"`
   - Tests reasoning and case application

4. **Case Summary**: `"Summarize the facts of the PCGG vs. Sandiganbayan case"`
   - Tests information extraction

5. **Complex Topic**: `"What is agrarian reform and how does it apply to reclassified lands?"`
   - Tests comprehensive multi-aspect analysis

## Output Format

### Test Results JSON Structure

```json
{
  "test_run_metadata": {
    "timestamp": "2025-01-17T14:30:22",
    "total_tests": 5,
    "environment": {...}
  },
  "summary_statistics": {
    "success_rate": "100%",
    "average_response_time_seconds": 2.345,
    ...
  },
  "test_results": [
    {
      "test_id": 1,
      "query": "What is the ruling...",
      "response": "Based on the case...",
      "response_time_seconds": 2.1,
      "success": true,
      ...
    }
  ]
}
```

### Analysis Report Sections

1. **Summary Statistics**: Overall performance metrics
2. **Response Quality Analysis**: Length distribution, quality indicators
3. **Analysis by Query Type**: Performance breakdown per query type
4. **Individual Test Results**: Detailed results for each test

## Metrics Tracked

### Performance Metrics
- Response time (min, max, average)
- Success rate
- Response length

### Quality Indicators
- Presence of GR numbers/case citations
- Use of legal terminology
- Structured formatting (numbered lists, sections)

## Customizing Tests

To add your own test queries, edit `test_chatbot_queries.py`:

```python
TEST_QUERIES = [
    {
        "id": 6,
        "query": "Your custom query here",
        "type": "custom_type",
        "description": "What this test validates"
    },
    # ... more queries
]
```

## Requirements

Make sure your backend services are running:

1. **Qdrant** (vector database)
   ```bash
   docker-compose up -d qdrant
   ```

2. **Django backend** (if applicable)
   ```bash
   python manage.py runserver
   ```

3. **Environment variables** configured in `.env`

## Troubleshooting

### Import Errors
Make sure you're running from the `backend` directory:
```bash
cd backend
python test_chatbot_queries.py
```

### Connection Errors
Check that Qdrant is running:
```bash
docker-compose ps
```

### No Results Directory
The script will automatically create `test_results/` on first run.

## Example Output

```
================================================================================
PHLaw Chatbot Testing Suite
================================================================================
Starting test run at: 2025-01-17 14:30:22
Total test cases: 5

================================================================================
Test #1: GR_NUMBER_LOOKUP
================================================================================
Query: What is the ruling in G.R. No. 161295?
Description: Direct GR number lookup - tests retrieval by case ID

Processing...

✓ Response generated in 2.1s

Response preview (first 500 chars):
Based on the case G.R. No. 161295 (PCGG vs. Sandiganbayan), the Supreme Court...

[... continues for all 5 tests ...]

================================================================================
TEST SUMMARY
================================================================================
Total Tests:           5
Successful:            5 (100.0%)
Failed:                0

Performance Metrics:
Avg Response Time:     2.345s
Min Response Time:     1.892s
Max Response Time:     3.102s

Response Quality:
Avg Response Length:   1247.2 chars
Min Response Length:   892 chars
Max Response Length:   1823 chars

================================================================================
✓ Results saved to: test_results/chatbot_test_20250117_143022.json
================================================================================
```

## Notes

- Test results are timestamped and never overwritten
- Each test run creates a new JSON file
- Analysis reports are saved alongside JSON files
- All output uses UTF-8 encoding for proper display of legal text

## Future Enhancements

Potential additions to the test suite:
- Conversational context testing (multi-turn dialogues)
- Citation accuracy validation
- Response hallucination detection
- Performance benchmarking across different embeddings
- A/B testing different retrieval strategies

