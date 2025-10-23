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

# Note: chat_engine imports removed to avoid import errors during Django test discovery

try:
    from chatbot.retriever import LegalRetriever
except ImportError as e:
    print(f"Warning: Could not import LegalRetriever: {e}")
    # Create mock class for testing
    class LegalRetriever:
        def __init__(self, collection="jurisprudence"):
            self.collection = collection
        def _clean_content(self, text):
            if not text:
                return ""
            # Simple cleaning for testing
            return text.replace("- N/A", "").replace("Supreme Court E-Library", "").strip()


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
            "recommendations": []
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
    
    def save_report(self, filename=None):
        """Save the test report to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results/chatbot_test_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
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


def run_comprehensive_chatbot_test():
    """Run comprehensive chatbot tests with real responses"""
    print("STARTING COMPREHENSIVE CHATBOT TEST WITH REAL RESPONSES")
    print("="*80)
    
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
    
    test_report = TestReportGenerator()
    test_report.start_test_run()
    test_report.start_category("Comprehensive Chatbot Tests (Case Digest Generation)", "Testing chatbot with 10 specific case digest generation queries from test_queries.json")
    
    # Test queries from test_queries.json - Case Digest & Summary Generation category
    test_queries = [
        {
            "query": "Make me a case digest of G.R. No. 217411.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 217411"
        },
        {
            "query": "Can you summarize the Supreme Court ruling in G.R. No. 200501?",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 200501"
        },
        {
            "query": "Provide the case digest for G.R. No. 201015.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 201015"
        },
        {
            "query": "Digest of the case with G.R. No. 201530.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 201530"
        },
        {
            "query": "I need the full case digest for G.R. No. 202045. Include the facts and issue.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 202045 with facts and issues"
        },
        {
            "query": "Summarize the Supreme Court decision in G.R. No. 202560.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 202560"
        },
        {
            "query": "Make me a case digest of G.R. No. 203075.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 203075"
        },
        {
            "query": "Provide the case digest for G.R. No. 203590.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 203590"
        },
        {
            "query": "Digest of the case with G.R. No. 204005.",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 204005"
        },
        {
            "query": "Can you summarize the Supreme Court ruling in G.R. No. 204520?",
            "expected_type": "digest",
            "description": "Case digest generation test for G.R. No. 204520"
        }
    ]

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
                
                test_report.add_test_result(
                    f"chatbot_test_{i+1}",
                    "passed",
                    f"Query {i+1} completed successfully",
                    1.0,
                    {
                        "query": test_case['query'],
                        "expected_type": test_case['expected_type'],
                        "description": test_case['description'],
                        "chatbot_response": chatbot_response,
                        "response_length": len(chatbot_response),
                        "status_code": response.status_code,
                        "response_analysis": response_analysis
                    }
                )
                
                print(f"Response: {chatbot_response[:200]}...")
                
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
    
    # Add performance metrics
    test_report.add_performance_metric("total_execution_time", test_report.test_results["summary"]["execution_time"])
    test_report.add_performance_metric("average_test_time", 
                                     test_report.test_results["summary"]["execution_time"] / test_report.test_results["summary"]["total_tests"])
    
    # Print summary
    test_report.print_summary()
    
    # Save report
    report_file = test_report.save_report()
    print(f"\nDetailed test report saved to: {report_file}")
    
    return report_file


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
        """Test case digest generation for all 10 G.R. numbers from test_queries.json"""
        test_queries = [
            "Make me a case digest of G.R. No. 217411.",
            "Can you summarize the Supreme Court ruling in G.R. No. 200501?",
            "Provide the case digest for G.R. No. 201015.",
            "Digest of the case with G.R. No. 201530.",
            "I need the full case digest for G.R. No. 202045. Include the facts and issue.",
            "Summarize the Supreme Court decision in G.R. No. 202560.",
            "Make me a case digest of G.R. No. 203075.",
            "Provide the case digest for G.R. No. 203590.",
            "Digest of the case with G.R. No. 204005.",
            "Can you summarize the Supreme Court ruling in G.R. No. 204520?"
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
    run_comprehensive_chatbot_test()