import os

from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'
    
    def ready(self):
        """Load essential models at Django startup (runs once under autoreloader)."""
        if os.environ.get("RUN_MAIN") != "true":
            return
        try:
            from .model_cache import (get_cached_bm25,
                                      get_cached_cross_encoder,
                                      get_cached_embedding_model)
            print("[LOADING] Loading models at startup...")
            # Preload embedding model and auxiliary models
            # Note: Local GGUF LLM is NOT loaded here - it's only used for contextual chunking
            # and will be lazy-loaded when needed by contextual_rag.py
            get_cached_embedding_model()
            get_cached_cross_encoder()
            get_cached_bm25()
            print("[SUCCESS] Essential models loaded at startup (GGUF LLM skipped - lazy load)")
        except Exception as e:
            print(f"[WARNING] Startup model load failed: {e}")