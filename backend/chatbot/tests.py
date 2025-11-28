import json
import os
import sys
import time
from datetime import datetime

# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Add the backend directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# CONFIGURATION - Edit these values to change test range
# =============================================================================
DEFAULT_START_ID = 1      # Starting query index (1-based)
DEFAULT_END_ID = 66       # Ending query index (1-based)
DEFAULT_JSON_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fast_test_queries.json")

# =============================================================================

# Note: chat_engine imports removed to avoid import errors during Django test discovery

from chatbot.evaluation_metrics import (AutomatedContentScoring,
                                        LegalAccuracyMetrics)
from chatbot.retriever import LegalRetriever


class TestReportGenerator:
    """Generate comprehensive JSON test reports for analysis"""
    
    def __init__(self, baseline_file=None):
        """
        Initialize test report generator.
        
        Args:
            baseline_file: Optional path to baseline results JSON file for Objective 3 comparison
        """
        self.baseline_file = baseline_file
        self.baseline_results = None
        if baseline_file and os.path.exists(baseline_file):
            self._load_baseline()
        
        self.test_results = {
            "test_run_info": {
                "timestamp": datetime.now().isoformat(),
                "test_suite": "PHLaw-Chatbot Case Digest Tests",
                "version": "1.0.0",
                "environment": {
                    "python_version": sys.version,
                    "platform": os.name,
                    "working_directory": os.getcwd()
                }
            },
            "test_categories": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "success_rate": 0.0,
                "execution_time": 0.0
            },
            "detailed_results": [],
            "performance_metrics": {},
            "recommendations": [],
            "evaluation_metrics": {
                "rouge_scores": {"rouge_1_f1": [], "rouge_2_f1": [], "rouge_l_f1": []},
                "bert_score_f1": [],
                "accuracy_scores": [],
                "content_relevance_scores": [],
                "hallucination_flags": [],
                "thesis_objectives": {
                    "objective_1_accuracy": {"target": "75-85%", "achieved": False, "score": 0.0},
                    "objective_2_content_relevance": {"target": "50-80%", "achieved": False, "score": 0.0},
                    "objective_3_hallucination_reduction": {"target": "15%", "achieved": False, "score": 0.0}
                }
            }
        }
        self.start_time = None
        self.current_category = None
    
    def _load_baseline(self):
        """Load baseline results from JSON file for comparison"""
        try:
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            
            # Extract baseline hallucination rate
            eval_metrics = baseline_data.get("evaluation_metrics", {})
            hallucination_flags = eval_metrics.get("hallucination_flags", [])
            
            if hallucination_flags:
                baseline_rate = (sum(1 for h in hallucination_flags if h) / len(hallucination_flags)) * 100
            else:
                baseline_rate = None  # No baseline data available
            
            self.baseline_results = {
                "hallucination_rate": baseline_rate,
                "timestamp": baseline_data.get("test_run_info", {}).get("timestamp", "Unknown"),
                "total_tests": baseline_data.get("summary", {}).get("total_tests", 0)
            }
            print(f"✓ Loaded baseline from {self.baseline_file}")
            if baseline_rate is not None:
                print(f"  Baseline hallucination rate: {baseline_rate:.2f}%")
        except Exception as e:
            print(f"⚠ Warning: Could not load baseline from {self.baseline_file}: {e}")
            self.baseline_results = None
    
    def save_baseline(self, filename=None):
        """Save current results as baseline for future comparisons"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result/baseline_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert numpy types and booleans to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, bool):
                return bool(obj)
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            return obj
        
        # Recursively convert all values
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(item) for item in d]
            else:
                return convert_types(d)
        
        converted_results = recursive_convert(self.test_results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Baseline saved to: {filename}")
        return filename
    
    def start_test_run(self):
        """Start timing the test run"""
        self.start_time = time.time()
    
    def end_test_run(self):
        """End timing and calculate summary"""
        if self.start_time:
            self.test_results["summary"]["execution_time"] = time.time() - self.start_time
        
        # Calculate success rate
        total = self.test_results["summary"]["total_tests"]
        if total > 0:
            passed = self.test_results["summary"]["passed"]
            self.test_results["summary"]["success_rate"] = (passed / total) * 100
    
    def start_category(self, category_name, description=""):
        """Start a new test category"""
        self.current_category = category_name
        self.test_results["test_categories"][category_name] = {
            "description": description,
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "error": 0
            }
        }
    
    def end_category(self):
        """End the current test category"""
        if self.current_category:
            # Calculate category summary
            category = self.test_results["test_categories"][self.current_category]
            tests = category["tests"]
            
            category["summary"]["total"] = len(tests)
            category["summary"]["passed"] = len([t for t in tests if t["status"] == "passed"])
            category["summary"]["failed"] = len([t for t in tests if t["status"] == "failed"])
            category["summary"]["error"] = len([t for t in tests if t["status"] == "error"])
            
            # Update overall summary
            self.test_results["summary"]["total_tests"] += category["summary"]["total"]
            self.test_results["summary"]["passed"] += category["summary"]["passed"]
            self.test_results["summary"]["failed"] += category["summary"]["failed"]
            self.test_results["summary"]["errors"] += category["summary"]["error"]
            
    def add_test_result(self, test_name, status, message, execution_time, details=None):
        """Add a test result to the current category"""
        if self.current_category:
            test_result = {
                "test_name": test_name,
                "status": status,
                "message": message,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "details": details or {}
            }
            
            self.test_results["test_categories"][self.current_category]["tests"].append(test_result)
            self.test_results["detailed_results"].append(test_result)
    
    def add_performance_metric(self, metric_name, value):
        """Add a performance metric"""
        self.test_results["performance_metrics"][metric_name] = value
    
    def add_evaluation_metrics(self, response, reference_text, case_metadata, query):
        """Add evaluation metrics for a test response"""
        try:
            # Calculate automated scores (passing query for section-aware evaluation)
            automated_scores = AutomatedContentScoring.score_legal_response(
                response, reference_text, original_query=query
            )
            
            # Calculate legal accuracy assessment
            accuracy_assessment = LegalAccuracyMetrics.assess_legal_information_accuracy(
                response, reference_text, query
            )
            
            # Extract ROUGE scores
            rouge_scores = automated_scores.get('rouge', {})
            self.test_results["evaluation_metrics"]["rouge_scores"]["rouge_1_f1"].append(rouge_scores.get('rouge_1', {}).get('f1', 0.0))
            self.test_results["evaluation_metrics"]["rouge_scores"]["rouge_2_f1"].append(rouge_scores.get('rouge_2', {}).get('f1', 0.0))
            self.test_results["evaluation_metrics"]["rouge_scores"]["rouge_l_f1"].append(rouge_scores.get('rouge_l', {}).get('f1', 0.0))
            
            # Extract BERTScore F1
            bert_f1 = automated_scores.get('bertscore', {}).get('f1', 0.0)
            self.test_results["evaluation_metrics"]["bert_score_f1"].append(bert_f1)

            # Add accuracy and relevance scores
            self.test_results["evaluation_metrics"]["accuracy_scores"].append(accuracy_assessment.get('accuracy_score', 0.0))
            self.test_results["evaluation_metrics"]["content_relevance_scores"].append(
                automated_scores.get('overall_relevance', {}).get('score', 0.0)
            )
            # Track hallucination flags (prefer LegalAccuracyMetrics if present; fallback to AutomatedContentScoring)
            hallucinated = None
            if isinstance(accuracy_assessment, dict):
                hallucinated = accuracy_assessment.get('hallucination_detected')
            if hallucinated is None and isinstance(automated_scores, dict):
                halluc_analysis = automated_scores.get('hallucination_analysis', {})
                hallucinated = halluc_analysis.get('hallucination_detected')
            if hallucinated is None:
                hallucinated = False
            self.test_results["evaluation_metrics"]["hallucination_flags"].append(bool(hallucinated))
            
            return {
                'automated_scores': automated_scores,
                'accuracy_assessment': accuracy_assessment
            }
            
        except Exception as e:
            print(f"Warning: Could not calculate evaluation metrics: {e}")
            return None
    
    def calculate_thesis_objectives(self):
        """Calculate thesis objective achievements"""
        eval_metrics = self.test_results["evaluation_metrics"]
        
        # Calculate averages
        avg_accuracy = sum(eval_metrics["accuracy_scores"]) / len(eval_metrics["accuracy_scores"]) if eval_metrics["accuracy_scores"] else 0.0
        avg_content_relevance = sum(eval_metrics["content_relevance_scores"]) / len(eval_metrics["content_relevance_scores"]) if eval_metrics["content_relevance_scores"] else 0.0
        
        # Convert to percentages
        accuracy_percentage = avg_accuracy * 100
        content_relevance_percentage = avg_content_relevance * 100
        
        # Check thesis objectives
        obj1_achieved = 75 <= accuracy_percentage <= 85
        obj2_achieved = 50 <= content_relevance_percentage <= 80
        
        # Calculate current hallucination rate
        halluc_rate = (sum(1 for h in eval_metrics.get("hallucination_flags", []) if h) / len(eval_metrics.get("hallucination_flags", [1]))) * 100 if eval_metrics.get("hallucination_flags") else 0.0
        
        # Objective 3: Hallucination reduction (need baseline comparison)
        baseline_rate = self.baseline_results.get("hallucination_rate") if self.baseline_results else None
        
        if baseline_rate is not None:
            # Calculate reduction from baseline
            reduction_percentage = baseline_rate - halluc_rate
            target_reduction = 25.0 - 15.0  # Target: reduce from 25% to 15% (10 percentage points)
            target_absolute_rate = 15.0
            
            # Objective 3 achieved if:
            # 1. Current rate is <= 15% AND
            # 2. Reduction from baseline is >= 10 percentage points
            obj3_achieved = halluc_rate <= target_absolute_rate and reduction_percentage >= target_reduction
            
            # Store reduction information
            eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["baseline_rate"] = round(baseline_rate, 2)
            eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["reduction_percentage"] = round(reduction_percentage, 2)
            eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["target_reduction"] = target_reduction
        else:
            # No baseline: just check if current rate is <= 15%
            obj3_achieved = halluc_rate <= 15.0
            eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["baseline_rate"] = None
            eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["reduction_percentage"] = None
            eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["note"] = "No baseline provided - only checking if rate <= 15%"
        
        # Update thesis objectives
        eval_metrics["thesis_objectives"]["objective_1_accuracy"]["achieved"] = obj1_achieved
        eval_metrics["thesis_objectives"]["objective_1_accuracy"]["score"] = round(accuracy_percentage, 2)
        
        eval_metrics["thesis_objectives"]["objective_2_content_relevance"]["achieved"] = obj2_achieved
        eval_metrics["thesis_objectives"]["objective_2_content_relevance"]["score"] = round(content_relevance_percentage, 2)
        
        eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["achieved"] = obj3_achieved
        eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["score"] = round(halluc_rate, 2)
        
        return {
            'accuracy_percentage': accuracy_percentage,
            'content_relevance_percentage': content_relevance_percentage,
            'hallucination_rate': halluc_rate,
            'baseline_rate': baseline_rate,
            'objectives_met': sum([obj1_achieved, obj2_achieved, obj3_achieved])
        }
    
    def print_summary(self):
        """Print a summary of test results"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        summary = self.test_results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Execution Time: {summary['execution_time']:.2f} seconds")
        
        # Print category summaries
        for category_name, category in self.test_results["test_categories"].items():
            print(f"\n{category_name}:")
            cat_summary = category["summary"]
            print(f"  Total: {cat_summary['total']}")
            print(f"  Passed: {cat_summary['passed']}")
            print(f"  Failed: {cat_summary['failed']}")
            print(f"  Error: {cat_summary['error']}")
        
        # Print thesis metrics summary
        self.print_thesis_metrics()
    
    def print_thesis_metrics(self):
        """Print thesis metrics summary"""
        print("\n" + "="*80)
        print("THESIS METRICS SUMMARY")
        print("="*80)
        
        eval_metrics = self.test_results["evaluation_metrics"]
        
        # Calculate averages
        rouge_1_avg = sum(eval_metrics["rouge_scores"]["rouge_1_f1"]) / len(eval_metrics["rouge_scores"]["rouge_1_f1"]) if eval_metrics["rouge_scores"]["rouge_1_f1"] else 0.0
        bert_f1_avg = sum(eval_metrics["bert_score_f1"]) / len(eval_metrics["bert_score_f1"]) if eval_metrics["bert_score_f1"] else 0.0
        accuracy_avg = sum(eval_metrics["accuracy_scores"]) / len(eval_metrics["accuracy_scores"]) if eval_metrics["accuracy_scores"] else 0.0
        relevance_avg = sum(eval_metrics["content_relevance_scores"]) / len(eval_metrics["content_relevance_scores"]) if eval_metrics["content_relevance_scores"] else 0.0
        
        print(f"\n📊 EVALUATION METRICS:")
        print(f"  Average ROUGE-1 Score: {rouge_1_avg:.4f}")
        print(f"  Average BERTScore F1: {bert_f1_avg:.4f}")
        print(f"  Average Accuracy: {accuracy_avg:.4f} ({accuracy_avg*100:.2f}%)")
        print(f"  Average Content Relevance: {relevance_avg:.4f} ({relevance_avg*100:.2f}%)")
        
        # Print year and category analysis
        print(f"\n📅 TEST DATA ANALYSIS:")
        print(f"  Year: 2005 (All 10 test cases)")
        print(f"  Category: Case Digest & Summary Generation (All 10 test cases)")
        print(f"  Case Types: G.R. No. (7), A.M. No. (2), A.C. No. (1)")
        
        # Print thesis objectives
        objectives = eval_metrics["thesis_objectives"]
        print(f"\n🎯 THESIS OBJECTIVES:")
        print(f"  Objective 1 (75-85% Accuracy): {'✓ ACHIEVED' if objectives['objective_1_accuracy']['achieved'] else '✗ NOT ACHIEVED'} ({objectives['objective_1_accuracy']['score']:.2f}%)")
        print(f"  Objective 2 (50-80% Content Relevance): {'✓ ACHIEVED' if objectives['objective_2_content_relevance']['achieved'] else '✗ NOT ACHIEVED'} ({objectives['objective_2_content_relevance']['score']:.2f}%)")
        
        # Objective 3 with baseline comparison
        obj3 = objectives['objective_3_hallucination_reduction']
        baseline_rate = obj3.get('baseline_rate')
        reduction = obj3.get('reduction_percentage')
        
        if baseline_rate is not None:
            print(f"  Objective 3 (Hallucination Reduction): {'✓ ACHIEVED' if obj3['achieved'] else '✗ NOT ACHIEVED'}")
            print(f"    Current Rate: {obj3['score']:.2f}%")
            print(f"    Baseline Rate: {baseline_rate:.2f}%")
            print(f"    Reduction: {reduction:.2f} percentage points (target: {obj3.get('target_reduction', 10.0):.1f}%)")
        else:
            print(f"  Objective 3 (Hallucination Reduction): {'✓ ACHIEVED' if obj3['achieved'] else '✗ NOT ACHIEVED'} (Current Rate: {obj3['score']:.2f}%)")
            print(f"    ⚠ No baseline provided - cannot calculate reduction. Use --baseline to compare against previous run.")
        
        print("\n" + "="*80)
    
    def save_report(self, filename=None):
        """Save the test report to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result/chatbot_test_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert numpy types and booleans to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, bool):
                return bool(obj)
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            return obj
        
        # Recursively convert all values
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_types(d)
        
        converted_results = recursive_convert(self.test_results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def save_csv_report(self, filename=None):
        """Save per-query scores as CSV for presentation"""
        import csv
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result/chatbot_test_{timestamp}.csv"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Extract test results for CSV
        test_results = []
        for category_name, category in self.test_results["test_categories"].items():
            for test in category["tests"]:
                details = test.get("details", {})
                evaluation_metrics = details.get("evaluation_metrics", {})
                
                # Extract automated scores if available
                automated_scores = evaluation_metrics.get("automated_scores", {})
                accuracy_assessment = evaluation_metrics.get("accuracy_assessment", {})
                
                row = {
                    'test_name': test.get("test_name", ""),
                    'status': test.get("status", ""),
                    'query': details.get("query", ""),
                    'year': details.get("year", ""),
                    'category': details.get("category", ""),
                    'case_number': details.get("case_number", ""),
                    'response_length': details.get("response_length", 0),
                    'execution_time': test.get("execution_time", 0.0),
                    'timestamp': test.get("timestamp", "")
                }
                
                # Add ROUGE scores
                if 'rouge' in automated_scores:
                    rouge_scores = automated_scores['rouge']
                    row.update({
                        'rouge_1_f1': rouge_scores.get('rouge_1', {}).get('f1', 0.0),
                        'rouge_2_f1': rouge_scores.get('rouge_2', {}).get('f1', 0.0),
                        'rouge_l_f1': rouge_scores.get('rouge_l', {}).get('f1', 0.0)
                    })
                else:
                    row.update({'rouge_1_f1': 0.0, 'rouge_2_f1': 0.0, 'rouge_l_f1': 0.0})

                # Add BERTScore F1
                row['bertscore_f1'] = automated_scores.get('bertscore', {}).get('f1', 0.0)
                
                # Add accuracy scores
                row.update({
                    'accuracy_score': accuracy_assessment.get('accuracy_score', 0.0),
                    'accuracy_level': accuracy_assessment.get('accuracy_level', ''),
                    'quality_threshold_met': accuracy_assessment.get('quality_threshold_met', False)
                })
                
                # Add content relevance
                if 'overall_relevance' in automated_scores:
                    row['content_relevance_score'] = automated_scores['overall_relevance'].get('score', 0.0)
                else:
                    row['content_relevance_score'] = 0.0
                
                test_results.append(row)
        
        # Define CSV headers
        headers = [
            'test_name', 'status', 'query', 'year', 'category', 'case_number',
            'response_length', 'execution_time', 'timestamp',
            'rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1',
            'bertscore_f1',
            'accuracy_score', 'accuracy_level', 'quality_threshold_met',
            'content_relevance_score'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for row in test_results:
                # Truncate long queries for CSV
                if len(row.get('query', '')) > 200:
                    row['query'] = row['query'][:200] + '...'
                writer.writerow(row)
        
        return filename


def load_test_queries_from_json(start_id=1, end_id=10, json_file=None):
    """Load test queries from backend/fast_test_queries.json (or specified file) based on index range (1-based)."""
    try:
        json_path = json_file or DEFAULT_JSON_FILE
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # New format produced by generate_test_queries.py
        all_queries = data.get("queries", [])
        # Select by 1-based indices
        start_idx = max(0, start_id - 1)
        end_idx_excl = min(len(all_queries), end_id)
        selected = all_queries[start_idx:end_idx_excl]

        filtered_queries = []
        for i, item in enumerate(selected, start=start_id):
            query_text = item.get("query", "")
            category = item.get("category", "")
            year = item.get("case_year", "")
            case_title = item.get("case_title", "")
            case_id = item.get("case_id", "")  # Include case_id for grouping
            gr = item.get("case_gr_number") or ""
            
            # Use case_gr_number as-is if available (already formatted in fast_test_queries.json)
            # Otherwise, try to construct from case_id if it looks like a case number
            if gr:
                case_number = gr  # Already properly formatted (e.g., "A.C. No. 11219", "G.R. No. 254248")
            elif case_id and ("G.R." in case_id or "A.C." in case_id or "A.M." in case_id):
                case_number = case_id  # Use case_id if it contains a case number format
            else:
                case_number = "Unknown"
            
            # Map expected type for analysis helpers
            cat_lower = category.lower()
            if "digest" in cat_lower:
                expected_type = "digest"
            elif "factual" in cat_lower or "q&a" in cat_lower or "qa" in cat_lower or "ruling" in query_text.lower():
                expected_type = "ruling"
            else:
                expected_type = "facts"

            # Extract ground_truth - can be string or dict with facts/issues/ruling
            ground_truth = item.get("ground_truth", "")
            
            filtered_queries.append({
                "query": query_text,
                "expected_type": expected_type,
                "description": f"{category} for {case_title} ({year})",
                "year": year,
                "category": category,
                "case_number": case_number,
                "case_id": case_id,  # Include case_id for proper grouping
                "case_title": case_title,  # Include case_title for reference
                "case_gr_number": gr,  # Include raw case_gr_number
                "ground_truth": ground_truth,  # Include ground_truth from JSON
                "query_index": i
            })

        print(f"📋 Loaded {len(filtered_queries)} queries (indexes {start_id}-{end_id}) from {json_path}")
        return filtered_queries
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {json_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {json_file}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading queries: {e}")
        return []


def run_comprehensive_chatbot_test(start_id=1, end_id=10, baseline_file=None, save_as_baseline=False, use_rag=True, json_file=None):
    """Run comprehensive chatbot tests with real responses
    
    Args:
        start_id: Starting query ID (1-based)
        end_id: Ending query ID (1-based)
        baseline_file: Optional path to baseline results JSON file for Objective 3 comparison
        save_as_baseline: If True, save current results as baseline after completion
        use_rag: If False, test without RAG (baseline mode - LLM only)
        json_file: Path to test queries JSON file (defaults to DEFAULT_JSON_FILE)
    """
    print("STARTING COMPREHENSIVE CHATBOT TEST WITH REAL RESPONSES")
    print("="*80)
    print(f"📊 Testing queries with IDs {start_id} to {end_id}")
    if not use_rag:
        print(f"🔧 BASELINE MODE: No RAG - LLM only")
        if not save_as_baseline:
            print(f"⚠️  Note: Consider using --save-baseline when running without RAG")
    if baseline_file:
        print(f"📋 Baseline file: {baseline_file}")
    if save_as_baseline:
        print(f"💾 Will save results as baseline after completion")
    
    # Setup Django for standalone execution
    import django
    from django.conf import settings

    # Add the backend directory to Python path
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    
    # Configure Django settings
    if not settings.configured:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
        django.setup()
    
    # Load test queries from JSON file
    query_json_file = json_file or DEFAULT_JSON_FILE
    test_queries = load_test_queries_from_json(start_id, end_id, query_json_file)
    
    if not test_queries:
        print("❌ No test queries loaded. Exiting.")
        return None, None
    
    test_report = TestReportGenerator(baseline_file=baseline_file)
    test_report.start_test_run()
    test_report.start_category(
        f"Comprehensive Chatbot Tests (Indexes {start_id}-{end_id})",
        f"Testing chatbot with {len(test_queries)} queries from test_queries.json"
    )

    try:
        from chatbot.views import ChatView
    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")
        return None
    
    # Group queries by case so we can preserve conversation history across the 3 queries
    from collections import OrderedDict
    grouped_by_case = OrderedDict()
    for item in test_queries:
        case_key = item.get('case_id') or item.get('case_number') or f"{item.get('case_title','Unknown')}|{item.get('year','') }"
        grouped_by_case.setdefault(case_key, []).append(item)

    case_counter = 0
    for case_key, case_queries in grouped_by_case.items():
        case_counter += 1
        history = []  # preserve for this case only
        
        for j, test_case in enumerate(case_queries):
            i = (case_counter, j+1)
            try:
                print(f"\nTesting Case {case_counter} Query {j+1}: {test_case['query']}")
                print(f"📜 History length: {len(history)} messages")
                
                # If baseline mode (no RAG), call chat_with_law_bot directly
                if not use_rag:
                    from chatbot.chat_engine import chat_with_law_bot
                    chatbot_response = chat_with_law_bot(test_case['query'], history=history, use_rag=False, template_mode="simplified")
                    response_status = 200  # Assume success if no exception
                else:
                    # Use DRF APIClient to properly test the API endpoint with history
                    from django.http import HttpRequest
                    from rest_framework.test import APIRequestFactory

                    # Create a mock request object with template_mode for testing
                    factory = APIRequestFactory()
                    request = factory.post('/api/chat/', {
                        'query': test_case['query'],
                        'history': history,
                        'template_mode': 'simplified'  # Use simplified template for tests
                    }, format='json')
                    
                    # Get the view and call it directly
                    view = ChatView.as_view()
                    try:
                        response = view(request)
                        response_status = response.status_code
                        
                        if response.status_code == 200:
                            # DRF Response should have .data attribute
                            if hasattr(response, 'data'):
                                data = response.data
                                chatbot_response = data.get("response", "")
                            else:
                                # Fallback for Django HttpResponse
                                chatbot_response = response.content.decode('utf-8') if hasattr(response, 'content') else ""
                        else:
                            chatbot_response = ""
                            # Handle both DRF Response and Django HttpResponse
                            error_msg = ""
                            if hasattr(response, 'data'):
                                error_msg = str(response.data)
                            elif hasattr(response, 'content'):
                                error_msg = response.content.decode('utf-8', errors='ignore')
                            else:
                                error_msg = str(response)
                            print(f"⚠️ API returned status {response_status}: {error_msg}")
                    except Exception as e:
                        response_status = 500
                        chatbot_response = ""
                        print(f"⚠️ Error calling view: {e}")
                        import traceback
                        traceback.print_exc()
                
                if response_status == 200:
                    # Append to history so the next turn keeps context
                    history.append({"role": "user", "content": test_case['query']})
                    history.append({"role": "assistant", "content": chatbot_response})
                    
                    # Use ground_truth directly from fast_test_queries.json
                    # Template: Facts, Issues, Ruling only
                    ground_truth = test_case.get('ground_truth', '')
                    if isinstance(ground_truth, dict):
                        # Format dict ground_truth (case digest format) into text
                        # Only include Facts, Issues, Ruling
                        parts = []
                        if 'facts' in ground_truth:
                            facts = ground_truth['facts']
                            if isinstance(facts, list):
                                parts.append(f"FACTS:\n{' '.join(facts)}")
                            else:
                                parts.append(f"FACTS:\n{facts}")
                        if 'issues' in ground_truth:
                            issues = ground_truth['issues']
                            if isinstance(issues, list):
                                parts.append(f"ISSUES:\n{'; '.join(issues)}")
                            else:
                                parts.append(f"ISSUES:\n{issues}")
                        if 'ruling' in ground_truth:
                            ruling = ground_truth['ruling']
                            if isinstance(ruling, list):
                                parts.append(f"RULING:\n{' '.join(ruling)}")
                            else:
                                parts.append(f"RULING:\n{ruling}")
                        reference_text = '\n\n'.join(parts) if parts else ''
                    elif isinstance(ground_truth, str):
                        reference_text = ground_truth
                    else:
                        # Fallback if ground_truth is missing or unexpected format
                        reference_text = f"Reference for {test_case.get('case_title','Unknown')} — ground truth not available."

                    # Minimal case_metadata for evaluation (case digest template: facts, issues, ruling only)
                    case_metadata = {
                        'case_title': test_case.get('case_title', f"Test Case {case_counter}"),
                        'gr_number': test_case.get('case_number', ''),
                        'year': test_case.get('year', '2023'),
                        'category': test_case.get('category', 'Case Digest & Summary Generation')
                    }
                    
                    # Calculate evaluation metrics
                    evaluation_results = test_report.add_evaluation_metrics(
                        chatbot_response, reference_text, case_metadata, test_case['query']
                    )
                    
                    test_report.add_test_result(
                        f"chatbot_test_case{case_counter}_q{j+1}",
                        "passed",
                        f"Case {case_counter} Query {j+1} completed successfully",
                        1.0,
                        {
                            "query": test_case['query'],
                            "expected_type": test_case['expected_type'],
                            "description": test_case['description'],
                            "year": test_case.get('year', '2023'),
                            "category": test_case.get('category', 'Case Digest & Summary Generation'),
                            "case_number": test_case.get('case_number', 'Unknown'),
                            "chatbot_response": chatbot_response,
                            "response_length": len(chatbot_response),
                            "status_code": response_status,
                            "mode": "baseline_no_rag" if not use_rag else "rag_enabled",
                            "evaluation_metrics": evaluation_results,
                            "case_id": case_key,
                            "turn_index": j+1
                        }
                    )
                    
                    print(f"[{test_case.get('year', '2023')}] [{test_case.get('category', 'Case Digest')}] {test_case.get('case_number', 'Unknown')}")
                    print(f"Response: {chatbot_response[:200]}...")
                    if evaluation_results:
                        print(f"Accuracy: {evaluation_results['accuracy_assessment'].get('accuracy_score',0.0):.4f}")
                    print("-" * 80)
                    
                else:
                    test_report.add_test_result(
                        f"chatbot_test_case{case_counter}_q{j+1}",
                        "failed",
                        f"Case {case_counter} Query {j+1} failed with status {response_status}",
                        1.0,
                        {
                            "query": test_case['query'],
                            "status_code": response_status,
                            "error": "HTTP error" if use_rag else "Generation error",
                            "case_id": case_key,
                            "turn_index": j+1,
                            "mode": "baseline_no_rag" if not use_rag else "rag_enabled"
                        }
                    )
                    print(f"Failed with status: {response_status}")
                
            except Exception as e:
                test_report.add_test_result(
                    f"chatbot_test_case{case_counter}_q{j+1}",
                    "error",
                    f"Case {case_counter} Query {j+1} caused an exception: {str(e)}",
                    1.0,
                    {
                        "query": test_case.get('query',''),
                        "error": str(e),
                        "exception_type": type(e).__name__,
                        "case_id": case_key,
                        "turn_index": j+1
                    }
                )
                print(f"Exception: {str(e)}")
    test_report.end_category()
    test_report.end_test_run()
    
    # Calculate thesis objectives
    thesis_results = test_report.calculate_thesis_objectives()
    
    # Add performance metrics
    test_report.add_performance_metric("total_execution_time", test_report.test_results["summary"]["execution_time"])
    test_report.add_performance_metric("average_test_time", 
                                     test_report.test_results["summary"]["execution_time"] / test_report.test_results["summary"]["total_tests"])
    
    # Add thesis metrics
    test_report.add_performance_metric("thesis_objectives_met", thesis_results['objectives_met'])
    test_report.add_performance_metric("accuracy_percentage", thesis_results['accuracy_percentage'])
    test_report.add_performance_metric("content_relevance_percentage", thesis_results['content_relevance_percentage'])
    
    # Print summary
    test_report.print_summary()
    
    # Save JSON report
    json_file = test_report.save_report()
    print(f"\nDetailed JSON report saved to: {json_file}")
    
    # Save CSV report
    csv_file = test_report.save_csv_report()
    print(f"CSV report saved to: {csv_file}")
    
    # Save as baseline if requested
    if save_as_baseline:
        baseline_file = test_report.save_baseline()
        print(f"\n💾 BASELINE SAVED:")
        print(f"  • Baseline file: {baseline_file}")
        print(f"  • Use this file with --baseline in future runs to calculate Objective 3 reduction")
    
    print(f"\n📊 PRESENTATION DATA READY:")
    print(f"  • JSON file: {json_file} (detailed analysis)")
    print(f"  • CSV file: {csv_file} (for Excel/Google Sheets)")
    
    return json_file, csv_file
# =============================================================================
# DJANGO TEST RUNNER COMPATIBILITY
# =============================================================================
# To run as Django test: python manage.py test chatbot.tests.ChatbotEvaluationTestCase
# To run standalone: python chatbot/tests.py --start 1 --end 10

class ChatbotEvaluationTestCase:
    """
    Django TestCase-compatible wrapper for the comprehensive chatbot evaluation.
    
    This allows the test to be discovered by Django's test runner:
        python manage.py test chatbot.tests.ChatbotEvaluationTestCase
        
    Or run standalone with command-line arguments:
        python chatbot/tests.py --start 1 --end 10 --no-rag --save-baseline
    """
    @staticmethod
    def test_comprehensive_evaluation():
        """Run comprehensive evaluation test - can be called by Django test runner"""
        return run_comprehensive_chatbot_test(
            DEFAULT_START_ID, DEFAULT_END_ID,
            baseline_file=None,
            save_as_baseline=False,
            use_rag=True
        )


if __name__ == "__main__":
    # Run comprehensive chatbot tests with real responses when script is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Run chatbot tests with configurable query range')
    parser.add_argument('--start', type=int, default=DEFAULT_START_ID, 
                       help=f'Starting query ID (default: {DEFAULT_START_ID})')
    parser.add_argument('--end', type=int, default=DEFAULT_END_ID, 
                       help=f'Ending query ID (default: {DEFAULT_END_ID})')
    parser.add_argument('--json', type=str, default=DEFAULT_JSON_FILE, 
                       help=f'Path to test queries JSON file (default: {DEFAULT_JSON_FILE})')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline results JSON file for Objective 3 comparison')
    parser.add_argument('--save-baseline', action='store_true',
                       help='Save current results as baseline for future comparisons')
    parser.add_argument('--no-rag', action='store_true',
                       help='Run without RAG (baseline mode - LLM only). Use this to generate baseline results.')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting tests with query IDs {args.start} to {args.end}")
    print(f"📁 Using JSON file: {args.json}")
    if args.no_rag:
        print(f"🔧 Running in BASELINE MODE (no RAG)")
        if not args.save_baseline:
            print(f"⚠️  Recommendation: Use --save-baseline with --no-rag to save baseline results")
    if args.baseline:
        print(f"📋 Baseline file: {args.baseline}")
    if args.save_baseline:
        print(f"💾 Will save as baseline")
    
    # Run the tests with custom JSON file
    run_comprehensive_chatbot_test(
        args.start, args.end, 
        baseline_file=args.baseline, 
        save_as_baseline=args.save_baseline,
        use_rag=not args.no_rag,
        json_file=args.json
    )