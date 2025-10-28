#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze chatbot test results from JSON files
Provides detailed analysis and quality metrics
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List


def load_test_results(json_file: str) -> Dict[str, Any]:
    """Load test results from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_response_quality(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze response quality metrics
    
    Args:
        results: List of test results
        
    Returns:
        Quality metrics dictionary
    """
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return {"error": "No successful tests to analyze"}
    
    # Response length analysis
    response_lengths = [r['response_length'] for r in successful_results]
    
    # Check for common quality indicators
    quality_indicators = {
        "has_case_citations": 0,
        "has_gr_numbers": 0,
        "has_legal_terms": 0,
        "has_structured_answer": 0
    }
    
    for result in successful_results:
        response = result['response'].lower()
        
        # Check for case citations
        if 'g.r. no.' in response or 'g.r. number' in response:
            quality_indicators['has_gr_numbers'] += 1
        
        # Check for legal terms
        legal_terms = ['petitioner', 'respondent', 'court', 'ruling', 'held', 'decided']
        if any(term in response for term in legal_terms):
            quality_indicators['has_legal_terms'] += 1
        
        # Check for structured content (numbered lists, sections)
        if any(marker in response for marker in ['1.', '2.', 'first', 'second', '**']):
            quality_indicators['has_structured_answer'] += 1
    
    return {
        "total_analyzed": len(successful_results),
        "response_length_stats": {
            "mean": sum(response_lengths) / len(response_lengths),
            "min": min(response_lengths),
            "max": max(response_lengths),
            "median": sorted(response_lengths)[len(response_lengths)//2]
        },
        "quality_indicators": {
            "percentage_with_gr_numbers": f"{(quality_indicators['has_gr_numbers']/len(successful_results))*100:.1f}%",
            "percentage_with_legal_terms": f"{(quality_indicators['has_legal_terms']/len(successful_results))*100:.1f}%",
            "percentage_with_structure": f"{(quality_indicators['has_structured_answer']/len(successful_results))*100:.1f}%"
        }
    }

def analyze_by_query_type(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results grouped by query type"""
    type_stats = {}
    
    for result in results:
        query_type = result['query_type']
        
        if query_type not in type_stats:
            type_stats[query_type] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "response_times": [],
                "response_lengths": []
            }
        
        type_stats[query_type]['total'] += 1
        
        if result['success']:
            type_stats[query_type]['successful'] += 1
            type_stats[query_type]['response_times'].append(result['response_time_seconds'])
            type_stats[query_type]['response_lengths'].append(result['response_length'])
        else:
            type_stats[query_type]['failed'] += 1
    
    # Calculate averages
    summary = {}
    for query_type, stats in type_stats.items():
        avg_time = sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else 0
        avg_length = sum(stats['response_lengths']) / len(stats['response_lengths']) if stats['response_lengths'] else 0
        
        summary[query_type] = {
            "total_queries": stats['total'],
            "success_rate": f"{(stats['successful']/stats['total'])*100:.1f}%",
            "avg_response_time": round(avg_time, 3),
            "avg_response_length": round(avg_length, 1)
        }
    
    return summary

def generate_detailed_report(data: Dict[str, Any]) -> str:
    """Generate a detailed text report"""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("DETAILED CHATBOT TEST ANALYSIS REPORT")
    report_lines.append("="*80)
    
    # Metadata
    metadata = data['test_run_metadata']
    report_lines.append(f"\nTest Run: {metadata['timestamp']}")
    report_lines.append(f"Total Tests: {metadata['total_tests']}")
    report_lines.append(f"Environment: Python {metadata['environment']['python_version']} on {metadata['environment']['platform']}")
    
    # Summary statistics
    summary = data['summary_statistics']
    report_lines.append("\n" + "-"*80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-"*80)
    report_lines.append(f"Success Rate:          {summary['success_rate']}")
    report_lines.append(f"Avg Response Time:     {summary['average_response_time_seconds']}s")
    report_lines.append(f"Response Time Range:   {summary['min_response_time_seconds']}s - {summary['max_response_time_seconds']}s")
    report_lines.append(f"Avg Response Length:   {summary['average_response_length']:.0f} characters")
    
    # Quality analysis
    quality = analyze_response_quality(data['test_results'])
    if 'error' not in quality:
        report_lines.append("\n" + "-"*80)
        report_lines.append("RESPONSE QUALITY ANALYSIS")
        report_lines.append("-"*80)
        report_lines.append(f"Responses Analyzed:    {quality['total_analyzed']}")
        report_lines.append(f"\nResponse Length Distribution:")
        report_lines.append(f"  Mean:     {quality['response_length_stats']['mean']:.1f} chars")
        report_lines.append(f"  Median:   {quality['response_length_stats']['median']} chars")
        report_lines.append(f"  Range:    {quality['response_length_stats']['min']} - {quality['response_length_stats']['max']} chars")
        report_lines.append(f"\nQuality Indicators:")
        for indicator, percentage in quality['quality_indicators'].items():
            indicator_name = indicator.replace('percentage_with_', '').replace('_', ' ').title()
            report_lines.append(f"  {indicator_name:.<40} {percentage:>10}")
    
    # Query type analysis
    type_analysis = analyze_by_query_type(data['test_results'])
    report_lines.append("\n" + "-"*80)
    report_lines.append("ANALYSIS BY QUERY TYPE")
    report_lines.append("-"*80)
    for query_type, stats in type_analysis.items():
        report_lines.append(f"\n{query_type.upper().replace('_', ' ')}:")
        report_lines.append(f"  Queries:       {stats['total_queries']}")
        report_lines.append(f"  Success Rate:  {stats['success_rate']}")
        report_lines.append(f"  Avg Time:      {stats['avg_response_time']}s")
        report_lines.append(f"  Avg Length:    {stats['avg_response_length']:.0f} chars")
    
    # Individual test details
    report_lines.append("\n" + "-"*80)
    report_lines.append("INDIVIDUAL TEST RESULTS")
    report_lines.append("-"*80)
    
    for result in data['test_results']:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        report_lines.append(f"\nTest #{result['test_id']}: {result['query_type']} - {status}")
        report_lines.append(f"Query: {result['query']}")
        report_lines.append(f"Response Time: {result['response_time_seconds']}s")
        
        if result['success']:
            report_lines.append(f"Response Length: {result['response_length']} characters")
            report_lines.append(f"\nResponse Preview (first 300 chars):")
            preview = result['response'][:300].replace('\n', ' ')
            report_lines.append(f"{preview}...")
        else:
            report_lines.append(f"Error: {result['error']}")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    return "\n".join(report_lines)

def main():
    """Main analysis function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_test_results.py <test_results.json>")
        print("\nOr to analyze the latest test result:")
        print("  python analyze_test_results.py latest")
        sys.exit(1)
    
    # Determine which file to analyze
    if sys.argv[1] == "latest":
        test_dir = os.path.join(os.path.dirname(__file__), "test_results")
        if not os.path.exists(test_dir):
            print(f"Error: Test results directory not found: {test_dir}")
            sys.exit(1)
        
        json_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
        if not json_files:
            print(f"Error: No test result files found in {test_dir}")
            sys.exit(1)
        
        # Get the latest file
        json_files.sort(reverse=True)
        input_file = os.path.join(test_dir, json_files[0])
        print(f"Analyzing latest test result: {json_files[0]}\n")
    else:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"Error: File not found: {input_file}")
            sys.exit(1)
    
    # Load and analyze results
    print("Loading test results...")
    data = load_test_results(input_file)
    
    print("Generating analysis report...\n")
    report = generate_detailed_report(data)
    
    # Print to console
    print(report)
    
    # Save report to file
    report_file = input_file.replace('.json', '_analysis.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Analysis report saved to: {report_file}\n")

if __name__ == "__main__":
    main()

