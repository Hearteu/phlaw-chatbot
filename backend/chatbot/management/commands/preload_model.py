from chatbot.generator import _ensure_llm
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Pre-load the LLM model for faster response times'

    def handle(self, *args, **options):
        self.stdout.write("🚀 Pre-loading LLM model...")
        try:
            llm = _ensure_llm()
            self.stdout.write(
                self.style.SUCCESS("✅ LLM model loaded successfully!")
            )
            self.stdout.write(f"   Model loaded and ready for use")
            self.stdout.write(f"   Model path: {llm.model_path}")
            
            # Test a simple generation to ensure model is working
            self.stdout.write("🧪 Testing model with simple query...")
            test_output = llm("Hello", max_tokens=10, temperature=0.1)
            if test_output and "choices" in test_output:
                self.stdout.write(
                    self.style.SUCCESS("✅ Model test successful!")
                )
            else:
                self.stdout.write(
                    self.style.WARNING("⚠️  Model loaded but test failed")
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Failed to load LLM model: {e}")
            )
