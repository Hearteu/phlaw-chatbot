# performance_monitor.py — Performance monitoring and optimization utilities
import functools
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    operation: str
    duration: float
    timestamp: float
    details: Dict[str, Any]
    success: bool


class PerformanceMonitor:
    """Performance monitoring system for contextual RAG operations"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: deque = deque(maxlen=max_history)
        self.lock = threading.Lock()
        self.stats = defaultdict(list)
        self.session_start = time.time()
    
    def record_metric(self, operation: str, duration: float, details: Dict[str, Any] = None, success: bool = True):
        """Record a performance metric"""
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=time.time(),
            details=details or {},
            success=success
        )
        
        with self.lock:
            self.metrics.append(metric)
            self.stats[operation].append(duration)
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for a specific operation"""
        if operation not in self.stats or not self.stats[operation]:
            return {}
        
        durations = self.stats[operation]
        return {
            'count': len(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations)
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics"""
        with self.lock:
            if not self.metrics:
                return {'total_operations': 0}
            
            total_duration = time.time() - self.session_start
            successful_ops = sum(1 for m in self.metrics if m.success)
            failed_ops = len(self.metrics) - successful_ops
            
            operation_stats = {}
            for operation in set(m.operation for m in self.metrics):
                operation_stats[operation] = self.get_operation_stats(operation)
            
            return {
                'session_duration': total_duration,
                'total_operations': len(self.metrics),
                'successful_operations': successful_ops,
                'failed_operations': failed_ops,
                'success_rate': successful_ops / len(self.metrics) if self.metrics else 0,
                'operation_stats': operation_stats
            }
    
    def get_slow_operations(self, threshold: float = 5.0) -> List[Dict[str, Any]]:
        """Get operations that took longer than threshold"""
        with self.lock:
            slow_ops = []
            for metric in self.metrics:
                if metric.duration > threshold:
                    slow_ops.append({
                        'operation': metric.operation,
                        'duration': metric.duration,
                        'timestamp': metric.timestamp,
                        'details': metric.details
                    })
            return slow_ops
    
    def export_metrics(self, filepath: str = "backend/data/performance_metrics.json"):
        """Export metrics to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with self.lock:
            export_data = {
                'session_stats': self.get_session_stats(),
                'slow_operations': self.get_slow_operations(),
                'all_metrics': [asdict(metric) for metric in self.metrics]
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Performance metrics exported to {filepath}")
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self.stats.clear()
            self.session_start = time.time()


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _performance_monitor


def monitor_performance(operation_name: str = None, details: Dict[str, Any] = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record successful operation
                monitor = get_performance_monitor()
                monitor.record_metric(
                    operation=operation,
                    duration=duration,
                    details=details or {},
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed operation
                monitor = get_performance_monitor()
                monitor.record_metric(
                    operation=operation,
                    duration=duration,
                    details={**(details or {}), 'error': str(e)},
                    success=False
                )
                
                raise
        
        return wrapper
    return decorator


class ContextualRAGProfiler:
    """Specialized profiler for contextual RAG operations"""
    
    def __init__(self):
        self.monitor = get_performance_monitor()
        self.active_timers = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.active_timers[operation] = time.time()
    
    def end_timer(self, operation: str, details: Dict[str, Any] = None, success: bool = True):
        """End timing an operation and record the metric"""
        if operation not in self.active_timers:
            print(f"⚠️ Timer for operation '{operation}' was not started")
            return
        
        duration = time.time() - self.active_timers[operation]
        del self.active_timers[operation]
        
        self.monitor.record_metric(
            operation=operation,
            duration=duration,
            details=details or {},
            success=success
        )
    
    def time_contextual_generation(self, num_chunks: int, method: str = "rule_based"):
        """Time contextual chunk generation"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timer(f"contextual_generation_{method}")
                
                try:
                    result = func(*args, **kwargs)
                    self.end_timer(
                        f"contextual_generation_{method}",
                        details={
                            'num_chunks': num_chunks,
                            'method': method,
                            'chunks_per_second': num_chunks / (time.time() - self.active_timers.get(f"contextual_generation_{method}", time.time()))
                        },
                        success=True
                    )
                    return result
                except Exception as e:
                    self.end_timer(
                        f"contextual_generation_{method}",
                        details={
                            'num_chunks': num_chunks,
                            'method': method,
                            'error': str(e)
                        },
                        success=False
                    )
                    raise
            
            return wrapper
        return decorator
    
    def time_retrieval(self, query: str, method: str = "hybrid"):
        """Time retrieval operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timer(f"retrieval_{method}")
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - self.active_timers.get(f"retrieval_{method}", time.time())
                    
                    self.end_timer(
                        f"retrieval_{method}",
                        details={
                            'query_length': len(query),
                            'method': method,
                            'results_count': len(result) if isinstance(result, list) else 0
                        },
                        success=True
                    )
                    return result
                except Exception as e:
                    self.end_timer(
                        f"retrieval_{method}",
                        details={
                            'query_length': len(query),
                            'method': method,
                            'error': str(e)
                        },
                        success=False
                    )
                    raise
            
            return wrapper
        return decorator


def print_performance_summary():
    """Print a summary of performance metrics"""
    monitor = get_performance_monitor()
    stats = monitor.get_session_stats()
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"Session Duration: {stats.get('session_duration', 0):.2f} seconds")
    print(f"Total Operations: {stats.get('total_operations', 0)}")
    print(f"Success Rate: {stats.get('success_rate', 0):.1%}")
    
    print("\nOperation Breakdown:")
    for operation, op_stats in stats.get('operation_stats', {}).items():
        print(f"  {operation}:")
        print(f"    Count: {op_stats.get('count', 0)}")
        print(f"    Avg Duration: {op_stats.get('avg_duration', 0):.3f}s")
        print(f"    Total Duration: {op_stats.get('total_duration', 0):.3f}s")
    
    # Show slow operations
    slow_ops = monitor.get_slow_operations(threshold=2.0)
    if slow_ops:
        print(f"\n🐌 Slow Operations (>2s):")
        for op in slow_ops[:5]:  # Show top 5
            print(f"  {op['operation']}: {op['duration']:.3f}s")
    
    print("="*60)


def optimize_based_on_metrics():
    """Provide optimization recommendations based on performance metrics"""
    monitor = get_performance_monitor()
    stats = monitor.get_session_stats()
    
    recommendations = []
    
    # Check for slow contextual generation
    contextual_stats = stats.get('operation_stats', {}).get('contextual_generation_llm', {})
    if contextual_stats.get('avg_duration', 0) > 5.0:
        recommendations.append({
            'issue': 'Slow contextual generation',
            'recommendation': 'Consider using more rule-based generation or reducing LLM calls',
            'impact': 'High'
        })
    
    # Check for slow retrieval
    retrieval_stats = stats.get('operation_stats', {}).get('retrieval_hybrid', {})
    if retrieval_stats.get('avg_duration', 0) > 3.0:
        recommendations.append({
            'issue': 'Slow retrieval operations',
            'recommendation': 'Reduce vector_k and bm25_k parameters or implement more caching',
            'impact': 'Medium'
        })
    
    # Check success rate
    if stats.get('success_rate', 1.0) < 0.9:
        recommendations.append({
            'issue': 'Low success rate',
            'recommendation': 'Review error handling and add more fallbacks',
            'impact': 'High'
        })
    
    if recommendations:
        print("\n🔧 OPTIMIZATION RECOMMENDATIONS:")
        print("="*50)
        for rec in recommendations:
            print(f"• {rec['issue']} ({rec['impact']} impact)")
            print(f"  → {rec['recommendation']}")
        print("="*50)
    else:
        print("\n✅ Performance looks good! No major issues detected.")
    
    return recommendations


# Convenience functions for common operations
def time_contextual_generation(func):
    """Decorator specifically for contextual generation timing"""
    return monitor_performance("contextual_generation", {'type': 'llm_call'})(func)


def time_retrieval(func):
    """Decorator specifically for retrieval timing"""
    return monitor_performance("retrieval", {'type': 'hybrid_search'})(func)


def time_embedding(func):
    """Decorator specifically for embedding timing"""
    return monitor_performance("embedding", {'type': 'vector_encoding'})(func)
