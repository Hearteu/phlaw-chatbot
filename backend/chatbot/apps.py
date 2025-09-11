import os

from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'
    
    def ready(self):
        """Full model load at Django startup (runs once under autoreloader)."""
        if os.environ.get("RUN_MAIN") != "true":
            return
        try:
            from .model_cache import (get_cached_bm25,
                                      get_cached_cross_encoder,
                                      get_cached_embedding_model,
                                      get_cached_llm)
            print("[LOADING] Loading models at startup...")
            get_cached_llm()
            get_cached_embedding_model()
            get_cached_cross_encoder()
            get_cached_bm25()
            print("[SUCCESS] Models loaded at startup")
        except Exception as e:
            print(f"[WARNING] Startup model load failed: {e}")