from chatbot.generator import _ensure_docker_model, generate_response
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Test Docker model connection for faster response times'

    def handle(self, *args, **options):
        self.stdout.write("🚀 Testing Docker model connection...")
        try:
            docker_client = _ensure_docker_model()
            if not docker_client.is_available:
                self.stdout.write(
                    self.style.ERROR("❌ Docker model not available")
                )
                return
                
            self.stdout.write(
                self.style.SUCCESS("✅ Docker model connection successful!")
            )
            self.stdout.write(f"   Model: {docker_client.model_name}")
            self.stdout.write(f"   Base URL: {docker_client.base_url}")
            
            # Test a simple generation to ensure model is working
            self.stdout.write("🧪 Testing model with simple query...")
            test_output = generate_response("Hello", max_tokens=10)
            if test_output and len(test_output.strip()) > 0:
                self.stdout.write(
                    self.style.SUCCESS("✅ Model test successful!")
                )
                self.stdout.write(f"   Test response: {test_output[:50]}...")
            else:
                self.stdout.write(
                    self.style.WARNING("⚠️  Model connected but test failed")
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Failed to connect to Docker model: {e}")
            )
