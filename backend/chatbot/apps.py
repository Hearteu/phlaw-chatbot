from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'
    
    def ready(self):
        """Initialize the LLM model when Django starts"""
        # Only load the model once when Django starts
        try:
            from .generator import _ensure_llm
            print("🚀 Pre-loading LLM model for Django...")
            llm = _ensure_llm()
            
            # Test the model with a simple query
            print("🧪 Testing model functionality...")
            test_output = llm("Test", max_tokens=5, temperature=0.1)
            if test_output and "choices" in test_output:
                print("✅ LLM model pre-loaded and tested successfully for Django")
            else:
                print("⚠️  LLM model loaded but test failed")
                
        except Exception as e:
            print(f"⚠️  Could not pre-load LLM model: {e}")
            print("   Model will be loaded on first request")
            print("   This is normal for the first startup")
