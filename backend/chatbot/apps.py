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
            from .docker_model_client import get_docker_model_client
            from .model_cache import (get_cached_bm25,
                                      get_cached_cross_encoder,
                                      get_cached_embedding_model)
            print("[LOADING] Loading models at startup (Docker-only)...")
            # Initialize Docker model client (tests availability and caches client)
            _ = get_docker_model_client()
            # Load auxiliary models (embedding, cross-encoder, BM25) as before
            get_cached_embedding_model()
            get_cached_cross_encoder()
            get_cached_bm25()
            print("[SUCCESS] Docker client and aux models loaded at startup")
        except Exception as e:
            print(f"[WARNING] Startup model load failed: {e}")