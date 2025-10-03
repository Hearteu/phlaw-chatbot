# model_cache.py — Enhanced centralized model caching with memory management
import gc
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import psutil
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

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
        print(f"⚠️ Memory cleanup failed: {e}")

@contextmanager
def memory_managed_operation():
    """Context manager for memory-managed operations"""
    memory_before = _monitor_memory()
    try:
        yield
    finally:
        memory_after = _monitor_memory()
        if _check_memory_pressure():
            print("⚠️ High memory usage detected, forcing cleanup...")
            _force_cleanup()

def get_cached_llm() -> Optional[Llama]:
    """Get cached LLM instance with enhanced memory management"""
    global _LLM_INSTANCE, _LLM_LOADED, _MEMORY_STATS
    
    with _CACHE_LOCK:
        if _LLM_INSTANCE is not None:
            # Test if the instance is still valid
            try:
                _ = _LLM_INSTANCE.model_path
                return _LLM_INSTANCE
            except Exception:
                # If the instance is corrupted, clear it and reload
                print("🔄 LLM instance corrupted, clearing cache...")
                clear_llm_cache()
    
    if not _LLM_LOADED:
        with memory_managed_operation():
            try:
                print("Loading LLM model...")
                start_time = time.time()
                
                # Check memory before loading
                memory_before = _monitor_memory()
                print(f"📊 Memory before LLM load: {memory_before.get('used_percent', 0):.1f}%")
                
                # LLM Configuration
                BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                MODEL_PATH = os.path.join(BASE_DIR, "law-chat.Q4_K_M.gguf")
                
                # Check if model file exists
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
                
                # Load-time configuration only; sampling is set at generation time
                LLM_CONFIG = {
                    "model_path": MODEL_PATH,
                    "n_ctx": 4096,
                    "n_gpu_layers": -1,
                    "n_threads": 8,
                    "n_batch": 256,
                    "n_ubatch": 16,
                    "use_mmap": True,
                    "use_mlock": False,
                    "low_vram": True,
                    "f16_kv": True,
                    "logits_all": False,
                    "embedding": False,
                    "verbose": False,
                }
            
                try:
                    _LLM_INSTANCE = Llama(**LLM_CONFIG)
                    load_time = time.time() - start_time
                    memory_after = _monitor_memory()
                    print(f"[SUCCESS] LLM model loaded in {load_time:.2f}s")
                    print(f"📊 Memory after LLM load: {memory_after.get('used_percent', 0):.1f}%")
                    _MEMORY_STATS["llm_loads"] += 1
                    _LLM_LOADED = True
                except Exception as e:
                    print(f"[ERROR] CUDA configuration failed: {e}")
                    print("🔄 Trying CPU-only fallback...")
                    
                    # Fallback to CPU-only configuration
                    cpu_config = {
                        "model_path": MODEL_PATH,
                        "n_ctx": 2048,
                        "n_gpu_layers": 0,
                        "n_threads": 4,
                        "n_batch": 16,
                        "use_mmap": True,
                        "use_mlock": False,
                        "f16_kv": True,
                        "logits_all": False,
                        "embedding": False,
                        "verbose": False,
                    }
                    
                    try:
                        _LLM_INSTANCE = Llama(**cpu_config)
                        load_time = time.time() - start_time
                        memory_after = _monitor_memory()
                        print(f"[SUCCESS] LLM model loaded in CPU-only mode in {load_time:.2f}s")
                        print(f"📊 Memory after LLM load: {memory_after.get('used_percent', 0):.1f}%")
                        _MEMORY_STATS["llm_loads"] += 1
                        _LLM_LOADED = True
                    except Exception as e2:
                        print(f"[ERROR] CPU fallback also failed: {e2}")
                        _LLM_LOADED = True  # Mark as loaded to prevent retry loops
                        return None
            
            except Exception as e:
                print(f"[ERROR] Failed to load LLM model: {e}")
                _LLM_LOADED = True  # Mark as loaded to prevent retry loops
                return None
    
    return _LLM_INSTANCE

def get_fresh_llm() -> Optional[Llama]:
    """Load a fresh Llama instance without touching or populating the global cache."""
    try:
        print("Loading fresh LLM (no cache)...")
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, "law-chat.Q4_K_M.gguf")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        cfg = {
            "model_path": MODEL_PATH,
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "n_batch": 512,
            "use_mmap": True,
            "use_mlock": False,
            "f16_kv": True,
            "logits_all": False,
            "embedding": False,
            "verbose": False,
        }
        try:
            return Llama(**cfg)
        except Exception:
            cpu_cfg = {
                "model_path": MODEL_PATH,
                "n_ctx": 2048,
                "n_gpu_layers": 0,
                "n_threads": 4,
                "n_batch": 16,
                "use_mmap": True,
                "use_mlock": False,
                "verbose": False,
            }
            return Llama(**cpu_cfg)
    except Exception as e:
        print(f"[ERROR] get_fresh_llm failed: {e}")
        return None

def clear_llm_cache():
    """Clear the LLM instance cache with enhanced memory management"""
    global _LLM_INSTANCE, _LLM_LOADED, _MEMORY_STATS
    
    with _CACHE_LOCK:
        memory_before = _monitor_memory()
        print(f"📊 Memory before LLM cache clear: {memory_before.get('used_percent', 0):.1f}%")
        
        if _LLM_INSTANCE is not None:
            try:
                # Try to properly close the model to avoid sampler attribute errors
                if hasattr(_LLM_INSTANCE, 'close'):
                    _LLM_INSTANCE.close()
            except Exception as e:
                # Ignore errors during cleanup, especially sampler attribute errors
                print(f"⚠️ Warning during LLM close: {e}")
            try:
                # Try to free the model from memory
                del _LLM_INSTANCE
            except Exception as e:
                # Ignore errors during deletion
                print(f"⚠️ Warning during LLM deletion: {e}")
        
        _LLM_INSTANCE = None
        _LLM_LOADED = False
        
        # Force garbage collection
        _force_cleanup()
        
        memory_after = _monitor_memory()
        print(f"📊 Memory after LLM cache clear: {memory_after.get('used_percent', 0):.1f}%")
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
# SENTENCE TRANSFORMER CACHING
# =============================================================================
def get_cached_embedding_model() -> Optional[SentenceTransformer]:
    """Get cached SentenceTransformer model with lazy loading"""
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    
    if _EMBEDDING_MODEL is None and not _EMBEDDING_MODEL_LOADED:
        print("🔄 Loading SentenceTransformer model...")
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
    print("[SUCCESS] All model caches cleared")

def get_cache_status() -> dict:
    """Get status of all model caches"""
    return {
        "llm_loaded": _LLM_INSTANCE is not None,
        "llm_marked_loaded": _LLM_LOADED,
        "embedding_loaded": _EMBEDDING_MODEL is not None,
        "embedding_marked_loaded": _EMBEDDING_MODEL_LOADED,
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
        print("⚠️ High memory pressure detected, performing automatic cleanup...")
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