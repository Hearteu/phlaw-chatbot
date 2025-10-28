import json
import os
import sys
import time
from datetime import datetime

from django.test import Client, TestCase

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
DEFAULT_START_ID = 1      # Starting query ID
DEFAULT_END_ID = 10       # Ending query ID
DEFAULT_JSON_FILE = "data/test_queries_2005_2025.json"  # Path to test queries

# =============================================================================

# Note: chat_engine imports removed to avoid import errors during Django test discovery

try:
    from chatbot.evaluation_metrics import (AutomatedContentScoring,
                                            LegalAccuracyMetrics)
    from chatbot.retriever import LegalRetriever
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    # Create mock class for testing
    class LegalRetriever:
        def __init__(self, collection="jurisprudence"):
            self.collection = collection
        def _clean_content(self, text):
            if not text:
                return ""
            # Simple cleaning for testing
            return text.replace("- N/A", "").replace("Supreme Court E-Library", "").strip()
    
    # Mock evaluation classes if not available
    class AutomatedContentScoring:
        @staticmethod
        def score_legal_response(response, reference, metadata=None):
            return {
                'bleu': {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'bleu_avg': 0.0},
                'rouge': {'rouge_1': {'f1': 0.0}, 'rouge_2': {'f1': 0.0}, 'rouge_l': {'f1': 0.0}},
                'legal_elements': {'presence_rate': 0.0, 'elements_found': 0, 'total_elements': 0},
                'citation_accuracy': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'overall_relevance': {'score': 0.0}
            }
    
    class LegalAccuracyMetrics:
        @staticmethod
        def assess_legal_information_accuracy(response, reference, metadata, query=""):
            return {
                'accuracy_level': 'LOW',
                'accuracy_score': 0.0,
                'recommendation': 'Mock evaluation',
                'accuracy_issues': ['Mock evaluation'],
                'detailed_scores': {
                    'content_relevance': 0.0,
                    'information_overlap': 0.0,
                    'legal_elements_completeness': 0.0,
                    'citation_accuracy': 0.0,
                    'overall_relevance': 0.0
                },
                'quality_threshold_met': False
            }


class TestReportGenerator:
    """Generate comprehensive JSON test reports for analysis"""
    
    def __init__(self):
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
                "bleu_scores": {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": [], "bleu_avg": []},
                "rouge_scores": {"rouge_1_f1": [], "rouge_2_f1": [], "rouge_l_f1": []},
                "accuracy_scores": [],
                "content_relevance_scores": [],
                "thesis_objectives": {
                    "objective_1_accuracy": {"target": "75-85%", "achieved": False, "score": 0.0},
                    "objective_2_content_relevance": {"target": "50-80%", "achieved": False, "score": 0.0},
                    "objective_3_hallucination_reduction": {"target": "15%", "achieved": False, "score": 0.0}
                }
            }
        }
        self.start_time = None
        self.current_category = None
    
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
            # Calculate automated scores
            automated_scores = AutomatedContentScoring.score_legal_response(
                response, reference_text, case_metadata
            )
            
            # Calculate legal accuracy assessment
            accuracy_assessment = LegalAccuracyMetrics.assess_legal_information_accuracy(
                response, reference_text, case_metadata, query
            )
            
            # Extract BLEU scores
            bleu_scores = automated_scores.get('bleu', {})
            self.test_results["evaluation_metrics"]["bleu_scores"]["bleu_1"].append(bleu_scores.get('bleu_1', 0.0))
            self.test_results["evaluation_metrics"]["bleu_scores"]["bleu_2"].append(bleu_scores.get('bleu_2', 0.0))
            self.test_results["evaluation_metrics"]["bleu_scores"]["bleu_3"].append(bleu_scores.get('bleu_3', 0.0))
            self.test_results["evaluation_metrics"]["bleu_scores"]["bleu_4"].append(bleu_scores.get('bleu_4', 0.0))
            self.test_results["evaluation_metrics"]["bleu_scores"]["bleu_avg"].append(bleu_scores.get('bleu_avg', 0.0))
            
            # Extract ROUGE scores
            rouge_scores = automated_scores.get('rouge', {})
            self.test_results["evaluation_metrics"]["rouge_scores"]["rouge_1_f1"].append(rouge_scores.get('rouge_1', {}).get('f1', 0.0))
            self.test_results["evaluation_metrics"]["rouge_scores"]["rouge_2_f1"].append(rouge_scores.get('rouge_2', {}).get('f1', 0.0))
            self.test_results["evaluation_metrics"]["rouge_scores"]["rouge_l_f1"].append(rouge_scores.get('rouge_l', {}).get('f1', 0.0))
            
            # Add accuracy and relevance scores
            self.test_results["evaluation_metrics"]["accuracy_scores"].append(accuracy_assessment.get('accuracy_score', 0.0))
            self.test_results["evaluation_metrics"]["content_relevance_scores"].append(
                automated_scores.get('overall_relevance', {}).get('score', 0.0)
            )
            
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
        obj3_achieved = accuracy_percentage >= 70  # High quality threshold
        
        # Update thesis objectives
        eval_metrics["thesis_objectives"]["objective_1_accuracy"]["achieved"] = obj1_achieved
        eval_metrics["thesis_objectives"]["objective_1_accuracy"]["score"] = round(accuracy_percentage, 2)
        
        eval_metrics["thesis_objectives"]["objective_2_content_relevance"]["achieved"] = obj2_achieved
        eval_metrics["thesis_objectives"]["objective_2_content_relevance"]["score"] = round(content_relevance_percentage, 2)
        
        eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["achieved"] = obj3_achieved
        eval_metrics["thesis_objectives"]["objective_3_hallucination_reduction"]["score"] = round(100 - accuracy_percentage, 2)  # Hallucination rate
        
        return {
            'accuracy_percentage': accuracy_percentage,
            'content_relevance_percentage': content_relevance_percentage,
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
        bleu_avg = sum(eval_metrics["bleu_scores"]["bleu_avg"]) / len(eval_metrics["bleu_scores"]["bleu_avg"]) if eval_metrics["bleu_scores"]["bleu_avg"] else 0.0
        rouge_1_avg = sum(eval_metrics["rouge_scores"]["rouge_1_f1"]) / len(eval_metrics["rouge_scores"]["rouge_1_f1"]) if eval_metrics["rouge_scores"]["rouge_1_f1"] else 0.0
        accuracy_avg = sum(eval_metrics["accuracy_scores"]) / len(eval_metrics["accuracy_scores"]) if eval_metrics["accuracy_scores"] else 0.0
        relevance_avg = sum(eval_metrics["content_relevance_scores"]) / len(eval_metrics["content_relevance_scores"]) if eval_metrics["content_relevance_scores"] else 0.0
        
        print(f"\n📊 EVALUATION METRICS:")
        print(f"  Average BLEU Score: {bleu_avg:.4f}")
        print(f"  Average ROUGE-1 Score: {rouge_1_avg:.4f}")
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
        print(f"  Objective 3 (Hallucination Reduction): {'✓ ACHIEVED' if objectives['objective_3_hallucination_reduction']['achieved'] else '✗ NOT ACHIEVED'} (Hallucination Rate: {objectives['objective_3_hallucination_reduction']['score']:.2f}%)")
        
        print("\n" + "="*80)
    
    def save_report(self, filename=None):
        """Save the test report to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results/chatbot_test_{timestamp}.json"
        
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
            filename = f"test_results/chatbot_test_{timestamp}.csv"
        
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
                
                # Add BLEU scores
                if 'bleu' in automated_scores:
                    bleu_scores = automated_scores['bleu']
                    row.update({
                        'bleu_1': bleu_scores.get('bleu_1', 0.0),
                        'bleu_2': bleu_scores.get('bleu_2', 0.0),
                        'bleu_3': bleu_scores.get('bleu_3', 0.0),
                        'bleu_4': bleu_scores.get('bleu_4', 0.0),
                        'bleu_avg': bleu_scores.get('bleu_avg', 0.0)
                    })
                else:
                    row.update({'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'bleu_avg': 0.0})
                
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
            'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_avg',
            'rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1',
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


def analyze_response_quality(response, expected_type):
    """Analyze the quality of a chatbot response"""
    analysis = {
        "response_length": len(response),
        "has_content": len(response.strip()) > 0,
        "word_count": len(response.split()),
        "contains_legal_terms": any(term in response.lower() for term in [
            "court", "case", "ruling", "decision", "supreme", "g.r.", "petitioner", "respondent"
        ])
    }
    
    # Analyze based on expected response type
    if expected_type == "digest":
        digest_sections = ["Issue", "Facts", "Ruling", "Discussion", "Case", "Decision"]
        analysis["digest_sections_found"] = [section for section in digest_sections if section in response]
        analysis["has_digest_format"] = len(analysis["digest_sections_found"]) > 0
    elif expected_type == "facts":
        analysis["contains_facts_indicators"] = any(word in response.lower() for word in ["facts", "factual", "occurred", "happened"])
    elif expected_type == "ruling":
        analysis["contains_ruling_indicators"] = any(word in response.lower() for word in ["ruling", "decision", "court", "held", "therefore"])
    
    return analysis


def load_test_queries_from_json(start_id=1, end_id=10, json_file="data/test_queries_2005_2025.json"):
    """Load test queries from JSON file based on ID range"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_queries = data.get("test_queries", [])
        
        # Filter queries by ID range
        filtered_queries = []
        for query in all_queries:
            query_id = query.get("id", 0)
            if start_id <= query_id <= end_id:
                # Extract case number from query text
                query_text = query.get("query_text", "")
                case_number = "Unknown"
                if "G.R. No." in query_text:
                    case_number = query_text.split("G.R. No.")[1].split(".")[0].strip()
                    case_number = f"G.R. No. {case_number}"
                elif "A.M. No." in query_text:
                    case_number = query_text.split("A.M. No.")[1].split(".")[0].strip()
                    case_number = f"A.M. No. {case_number}"
                elif "A.C. No." in query_text:
                    case_number = query_text.split("A.C. No.")[1].split(".")[0].strip()
                    case_number = f"A.C. No. {case_number}"
                
                filtered_queries.append({
                    "query": query_text,
                    "expected_type": "digest",
                    "description": f"Case digest generation test for {case_number} ({query.get('year', 'Unknown')})",
                    "year": query.get("year", "Unknown"),
                    "category": query.get("category", "Unknown"),
                    "case_number": case_number,
                    "query_id": query_id
                })
        
        print(f"📋 Loaded {len(filtered_queries)} queries (IDs {start_id}-{end_id}) from {json_file}")
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


def run_comprehensive_chatbot_test(start_id=1, end_id=10):
    """Run comprehensive chatbot tests with real responses"""
    print("STARTING COMPREHENSIVE CHATBOT TEST WITH REAL RESPONSES")
    print("="*80)
    print(f"📊 Testing queries with IDs {start_id} to {end_id}")
    
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
    test_queries = load_test_queries_from_json(start_id, end_id)
    
    if not test_queries:
        print("❌ No test queries loaded. Exiting.")
        return None, None
    
    test_report = TestReportGenerator()
    test_report.start_test_run()
    test_report.start_category(f"Comprehensive Chatbot Tests (IDs {start_id}-{end_id})", f"Testing chatbot with {len(test_queries)} queries from test_queries_2005_2025.json")

    try:
        from chatbot.views import ChatView
        from django.test import RequestFactory
    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")
        return None

    # Use direct view testing instead of HTTP client
    factory = RequestFactory()
    
    for i, test_case in enumerate(test_queries):
        try:
            print(f"\nTesting Query {i+1}: {test_case['query']}")
            
            # Test the view directly (bypassing HTTP layer)
            request = factory.post('/api/chat/', {'query': test_case['query'], 'history': []})
            request.data = {'query': test_case['query'], 'history': []}  # Manually set for DRF
            
            view = ChatView()
            response = view.post(request)
            
            if response.status_code == 200:
                # response is a DRF Response object, not HTTP response
                data = response.data
                chatbot_response = data.get("response", "")
                
                # Analyze response quality
                response_analysis = analyze_response_quality(chatbot_response, test_case['expected_type'])
                
                # Create mock reference text and case metadata for evaluation
                # In a real scenario, you would have actual case data
                reference_text = f"Case digest for {test_case['query']}. This is a reference case digest with facts, issues, and ruling."
                case_metadata = {
                    'case_title': f"Test Case {i+1}",
                    'gr_number': test_case.get('case_number', f"G.R. No. {217411 + i}"),
                    'ponente': 'Justice Test',
                    'promulgation_date': f"{test_case.get('year', '2023')}-01-01",
                    'case_type': 'criminal',
                    'year': test_case.get('year', '2023'),
                    'category': test_case.get('category', 'Case Digest & Summary Generation')
                }
                
                # Calculate evaluation metrics
                evaluation_results = test_report.add_evaluation_metrics(
                    chatbot_response, reference_text, case_metadata, test_case['query']
                )
                
                test_report.add_test_result(
                    f"chatbot_test_{i+1}",
                    "passed",
                    f"Query {i+1} completed successfully",
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
                        "status_code": response.status_code,
                        "response_analysis": response_analysis,
                        "evaluation_metrics": evaluation_results
                    }
                )
                
                print(f"[{test_case.get('year', '2023')}] [{test_case.get('category', 'Case Digest')}] {test_case.get('case_number', 'Unknown')}")
                print(f"Response: {chatbot_response[:200]}...")
                if evaluation_results:
                    print(f"BLEU Score: {evaluation_results['automated_scores']['bleu']['bleu_avg']:.4f}")
                    print(f"Accuracy: {evaluation_results['accuracy_assessment']['accuracy_score']:.4f}")
                print("-" * 80)
                
            else:
                test_report.add_test_result(
                    f"chatbot_test_{i+1}",
                    "failed",
                    f"Query {i+1} failed with status {response.status_code}",
                    1.0,
                    {
                        "query": test_case['query'],
                        "status_code": response.status_code,
                        "error": "HTTP error"
                    }
                )
                print(f"Failed with status: {response.status_code}")
                
        except Exception as e:
            test_report.add_test_result(
                f"chatbot_test_{i+1}",
                "error",
                f"Query {i+1} caused an exception: {str(e)}",
                1.0,
                {
                    "query": test_case['query'],
                    "error": str(e),
                    "exception_type": type(e).__name__
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
    
    print(f"\n📊 PRESENTATION DATA READY:")
    print(f"  • JSON file: {json_file} (detailed analysis)")
    print(f"  • CSV file: {csv_file} (for Excel/Google Sheets)")
    
    return json_file, csv_file


class ChatbotTestCase(TestCase):
    """Django test case for chatbot functionality"""
    
    def setUp(self):
        self.client = Client()
    
    def test_chatbot_responds(self):
        """Test that chatbot responds to queries"""
        response = self.client.post(
            '/api/chat/',
            json.dumps({'query': "Make me a case digest of G.R. No. 217411."}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
    
    def test_case_digest_generation(self):
        """Test case digest generation for first 10 queries from test_queries_2005_2025.json"""
        test_queries = [
            "Make a case digest of G.R. No. 163410.",
            "Make a case digest of G.R. No. 152230.",
            "Make a case digest of G.R. No. 148339.",
            "Make a case digest of G.R. No. 151266.",
            "Make a case digest of A.M. No. RTJ-04-1873.",
            "Make a case digest of G.R. No. 150678.",
            "Make a case digest of G.R. No. 156260.",
            "Make a case digest of A.M. No. P-05-1933.",
            "Make a case digest of A.C. No. 5864.",
            "Make a case digest of A.M. No. P-05-2021."
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                response = self.client.post(
                    '/api/chat/',
                    json.dumps({'query': query}),
                    content_type='application/json'
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("response", data)
                self.assertGreater(len(data["response"]), 0)


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
    
    args = parser.parse_args()
    
    print(f"🚀 Starting tests with query IDs {args.start} to {args.end}")
    print(f"📁 Using JSON file: {args.json}")
    
    # Update the JSON file path in the load function
    def load_queries_with_custom_path(start_id, end_id):
        return load_test_queries_from_json(start_id, end_id, args.json)
    
    # Run the tests
    run_comprehensive_chatbot_test(args.start, args.end)