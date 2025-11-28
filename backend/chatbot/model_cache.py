# model_cache.py — Enhanced centralized model caching with memory management
import gc
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import psutil
from sentence_transformers import SentenceTransformer

from .docker_model_client import DockerModelClient, get_docker_model_client

# Optional heavy deps imported lazily in functions

# =============================================================================
# GLOBAL MODEL CACHES WITH MEMORY MANAGEMENT
# =============================================================================
_LLM_INSTANCE = None
_LLM_LOADED = False
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_LOADED = False
_CROSS_ENCODER_INSTANCE = None
_CROSS_ENCODER_LOADED = False
_BM25_MODEL = None
_BM25_CORPUS: Optional[List[List[str]]] = None
_BM25_DOC_METADATA: Optional[List[Dict[str, Any]]] = None
_BM25_LOADED = False
_MEMORY_MONITOR = None
_MEMORY_THRESHOLD = 0.85  # 85% memory usage threshold
_CACHE_LOCK = threading.Lock()
_MEMORY_STATS = {
    "llm_loads": 0,
    "embedding_loads": 0,
    "cross_encoder_loads": 0,
    "bm25_loads": 0,
    "memory_cleanups": 0,
    "last_cleanup": 0
}

# Base directory for data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENABLE_DOCKER_LLM = os.getenv("ENABLE_DOCKER_LLM", "false").lower() == "true"


class DockerLlamaAdapter:
    """Adapter exposing Docker-hosted Llama 3.2 through a llama.cpp-like interface."""

    def __init__(self, docker_client: DockerModelClient):
        self._client = docker_client
        self.model_path = f"{docker_client.model_name}@{docker_client.base_url}"

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.85,
        stop: Optional[List[str]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        call_kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            call_kwargs["stop"] = stop

        response = self._client.generate_response(prompt, **call_kwargs)
        return {
            "id": "docker-llama-3.2",
            "choices": [{"text": response}],
        }

    def close(self) -> None:
        """Compatibility no-op."""
        return None

# =============================================================================
# LLM CACHING
# =============================================================================
def _monitor_memory() -> Dict[str, float]:
    """Monitor system memory usage"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_percent": memory.percent,
            "free_gb": memory.free / (1024**3)
        }
    except Exception:
        return {"error": "Unable to monitor memory"}

def _check_memory_pressure() -> bool:
    """Check if memory pressure requires cleanup"""
    memory_info = _monitor_memory()
    if "error" in memory_info:
        return False
    return memory_info["used_percent"] > (_MEMORY_THRESHOLD * 100)

def _force_cleanup():
    """Force garbage collection and memory cleanup"""
    global _MEMORY_STATS
    try:
        gc.collect()
        _MEMORY_STATS["memory_cleanups"] += 1
        _MEMORY_STATS["last_cleanup"] = time.time()
        print("🧹 Memory cleanup completed")
    except Exception as e:
        print(f"Memory cleanup failed: {e}")

@contextmanager
def memory_managed_operation():
    """Context manager for memory-managed operations"""
    memory_before = _monitor_memory()
    try:
        yield
    finally:
        memory_after = _monitor_memory()
        if _check_memory_pressure():
            print("High memory usage detected, forcing cleanup...")
            _force_cleanup()

def get_cached_llm() -> Optional[Any]:
    """Get cached Docker-hosted LLM instance with enhanced memory management"""
    global _LLM_INSTANCE, _LLM_LOADED, _MEMORY_STATS

    if not ENABLE_DOCKER_LLM:
        with _CACHE_LOCK:
            if not _LLM_LOADED:
                print("ℹ️ Docker LLM disabled via configuration; returning None")
            _LLM_INSTANCE = None
            _LLM_LOADED = True
        return None
    
    with _CACHE_LOCK:
        if _LLM_INSTANCE is not None:
            # Test if the instance is still valid
            try:
                _ = _LLM_INSTANCE.model_path
                return _LLM_INSTANCE
            except Exception:
                # If the instance is corrupted, clear it and reload
                print("LLM instance corrupted, clearing cache...")
                clear_llm_cache()
    
    if not _LLM_LOADED:
        with memory_managed_operation():
            try:
                print("Initializing Docker-hosted Llama 3.2 client...")
                start_time = time.time()

                # Check memory before loading (still useful for monitoring)
                memory_before = _monitor_memory()
                print(f"Memory before LLM init: {memory_before.get('used_percent', 0):.1f}%")

                docker_client = get_docker_model_client()
                if not docker_client.is_available and hasattr(docker_client, "_test_connection"):
                    docker_client._test_connection()

                if not docker_client.is_available:
                    raise RuntimeError("Docker model runner not available")

                _LLM_INSTANCE = DockerLlamaAdapter(docker_client)

                load_time = time.time() - start_time
                memory_after = _monitor_memory()
                print(f"[SUCCESS] Docker Llama 3.2 client ready in {load_time:.2f}s")
                print(f"Memory after LLM init: {memory_after.get('used_percent', 0):.1f}%")
                _MEMORY_STATS["llm_loads"] += 1
                _LLM_LOADED = True
            except Exception as e:
                print(f"[ERROR] Failed to initialize Docker Llama 3.2 client: {e}")
                _LLM_INSTANCE = None
                _LLM_LOADED = False
                return None
    
    return _LLM_INSTANCE

def get_fresh_llm() -> Optional[Any]:
    """Load a fresh Docker Llama 3.2 adapter without touching the global cache."""
    try:
        if not ENABLE_DOCKER_LLM:
            print("ℹ️ Docker LLM disabled via configuration; fresh instance unavailable")
            return None
        print("Initializing fresh Docker Llama 3.2 client (no cache)...")
        docker_client = get_docker_model_client()
        if not docker_client.is_available and hasattr(docker_client, "_test_connection"):
            docker_client._test_connection()

        if not docker_client.is_available:
            raise RuntimeError("Docker model runner not available")

        return DockerLlamaAdapter(docker_client)
    except Exception as e:
        print(f"[ERROR] get_fresh_llm failed: {e}")
        return None

def clear_llm_cache():
    """Clear the LLM instance cache with enhanced memory management"""
    global _LLM_INSTANCE, _LLM_LOADED, _MEMORY_STATS
    
    with _CACHE_LOCK:
        memory_before = _monitor_memory()
        print(f"Memory before LLM cache clear: {memory_before.get('used_percent', 0):.1f}%")
        
        if _LLM_INSTANCE is not None:
            try:
                # Try to properly close the model to avoid sampler attribute errors
                if hasattr(_LLM_INSTANCE, 'close'):
                    _LLM_INSTANCE.close()
            except Exception as e:
                # Ignore errors during cleanup, especially sampler attribute errors
                print(f"Warning during LLM close: {e}")
            try:
                # Try to free the model from memory
                del _LLM_INSTANCE
            except Exception as e:
                # Ignore errors during deletion
                print(f"Warning during LLM deletion: {e}")
        
        _LLM_INSTANCE = None
        _LLM_LOADED = False
        
        # Force garbage collection
        _force_cleanup()
        
        memory_after = _monitor_memory()
        print(f"Memory after LLM cache clear: {memory_after.get('used_percent', 0):.1f}%")
        print("[CLEARED] LLM cache cleared with memory management")

def reload_llm():
    """Reload LLM by clearing cache and loading a fresh instance."""
    clear_llm_cache()
    return get_cached_llm()

# =============================================================================
# CROSS-ENCODER CACHING
# =============================================================================
def get_cached_cross_encoder():
    """Cross-encoder disabled (vector-only mode)."""
    _ = (_CROSS_ENCODER_INSTANCE, _CROSS_ENCODER_LOADED)  # keep globals referenced
    print("ℹ️ Cross-encoder caching disabled")
    return None

def clear_cross_encoder_cache():
    """No-op: Cross-encoder disabled."""
    print("ℹ️ Cross-encoder cache clear skipped (disabled)")

# =============================================================================
# BM25 CACHING
# =============================================================================
def get_cached_bm25() -> Tuple[Optional[Any], Optional[List[List[str]]], Optional[List[Dict[str, Any]]]]:
    """BM25 disabled (vector-only mode)."""
    print("ℹ️ BM25 caching disabled")
    return None, [], []

def clear_bm25_cache():
    """No-op: BM25 disabled."""
    print("ℹ️ BM25 cache clear skipped (disabled)")

# =============================================================================
# LEGAL DOCUMENT CLASSIFIER CACHING
# =============================================================================
_LEGAL_CLASSIFIER = None
_LEGAL_CLASSIFIER_LOADED = False

def get_cached_legal_classifier():
    """Get cached Saibo legal document classifier with lazy loading"""
    global _LEGAL_CLASSIFIER, _LEGAL_CLASSIFIER_LOADED
    
    if _LEGAL_CLASSIFIER is None and not _LEGAL_CLASSIFIER_LOADED:
        try:
            from .legal_document_classifier import \
                get_legal_document_classifier
            print("Loading Saibo legal RoBERTa document classifier...")
            start_time = time.time()
            
            _LEGAL_CLASSIFIER = get_legal_document_classifier()
            
            load_time = time.time() - start_time
            print(f"[SUCCESS] Legal document classifier loaded in {load_time:.2f}s")
            _LEGAL_CLASSIFIER_LOADED = True
        except Exception as e:
            print(f"Failed to load legal document classifier: {e}")
            _LEGAL_CLASSIFIER = None
            _LEGAL_CLASSIFIER_LOADED = True  # Mark as attempted to avoid retries
    
    return _LEGAL_CLASSIFIER

def clear_legal_classifier_cache():
    """Clear the legal document classifier cache to free memory"""
    global _LEGAL_CLASSIFIER, _LEGAL_CLASSIFIER_LOADED
    if _LEGAL_CLASSIFIER is not None:
        try:
            # Try to free the classifier from memory
            del _LEGAL_CLASSIFIER
        except Exception as e:
            # Ignore errors during cleanup
            pass
    _LEGAL_CLASSIFIER = None
    _LEGAL_CLASSIFIER_LOADED = False
    print("[CLEARED] Legal document classifier cache cleared")

# =============================================================================
# SENTENCE TRANSFORMER CACHING
# =============================================================================
def get_cached_embedding_model() -> Optional[SentenceTransformer]:
    """Get cached SentenceTransformer model with lazy loading"""
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    
    if _EMBEDDING_MODEL is None and not _EMBEDDING_MODEL_LOADED:
        print("Loading SentenceTransformer model...")
        start_time = time.time()
        
        # Use optimized model path
        model_name = os.getenv("EMBED_MODEL", "Stern5497/sbert-legal-xlm-roberta-base")
        device = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
        
        _EMBEDDING_MODEL = SentenceTransformer(model_name, device=device)
        
        load_time = time.time() - start_time
        print(f"[SUCCESS] SentenceTransformer model loaded in {load_time:.2f}s")
        _EMBEDDING_MODEL_LOADED = True
    
    return _EMBEDDING_MODEL

def clear_embedding_model_cache():
    """Clear the SentenceTransformer model cache to free memory"""
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    if _EMBEDDING_MODEL is not None:
        try:
            # Try to free the model from memory
            del _EMBEDDING_MODEL
        except Exception as e:
            # Ignore errors during cleanup
            pass
    _EMBEDDING_MODEL = None
    _EMBEDDING_MODEL_LOADED = False
    print("[CLEARED] SentenceTransformer model cache cleared")

# Backward-compatible alias used by older modules
def clear_embedding_cache():
    """Alias for backward compatibility."""
    return clear_embedding_model_cache()

# =============================================================================
# COMBINED CACHE MANAGEMENT
# =============================================================================
def clear_all_model_caches():
    """Clear all model caches to free memory"""
    print("[CLEARING] All model caches...")
    clear_llm_cache()
    clear_embedding_model_cache()
    clear_legal_classifier_cache()
    print("[SUCCESS] All model caches cleared")

def get_cache_status() -> dict:
    """Get status of all model caches"""
    return {
        "llm_loaded": _LLM_INSTANCE is not None,
        "llm_marked_loaded": _LLM_LOADED,
        "embedding_loaded": _EMBEDDING_MODEL is not None,
        "embedding_marked_loaded": _EMBEDDING_MODEL_LOADED,
        "legal_classifier_loaded": _LEGAL_CLASSIFIER is not None,
        "legal_classifier_marked_loaded": _LEGAL_CLASSIFIER_LOADED,
    }

def print_cache_status():
    """Print current cache status with memory information"""
    status = get_cache_status()
    memory_info = _monitor_memory()
    
    print("MODEL CACHE STATUS:")
    print(f"  LLM Model: {'[LOADED]' if status['llm_loaded'] else '[NOT LOADED]'}")
    print(f"  Embedding Model: {'[LOADED]' if status['embedding_loaded'] else '[NOT LOADED]'}")
    print(f"  LLM Marked Loaded: {status['llm_marked_loaded']}")
    print(f"  Embedding Marked Loaded: {status['embedding_marked_loaded']}")
    print(f"  Memory Usage: {memory_info.get('used_percent', 0):.1f}%")
    print(f"  Available Memory: {memory_info.get('available_gb', 0):.1f} GB")
    print(f"  Memory Pressure: {'HIGH' if _check_memory_pressure() else 'NORMAL'}")

def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics"""
    memory_info = _monitor_memory()
    return {
        **memory_info,
        "memory_pressure": _check_memory_pressure(),
        "cache_stats": _MEMORY_STATS.copy(),
        "threshold": _MEMORY_THRESHOLD
    }

def auto_cleanup_if_needed():
    """Automatically cleanup if memory pressure is high"""
    if _check_memory_pressure():
        print("High memory pressure detected, performing automatic cleanup...")
        clear_all_model_caches()
        return True
    return False

def set_memory_threshold(threshold: float):
    """Set memory threshold for automatic cleanup (0.0 to 1.0)"""
    global _MEMORY_THRESHOLD
    if 0.0 <= threshold <= 1.0:
        _MEMORY_THRESHOLD = threshold
        print(f"Memory threshold set to {threshold * 100:.1f}%")
    else:
        print("Invalid threshold. Must be between 0.0 and 1.0")