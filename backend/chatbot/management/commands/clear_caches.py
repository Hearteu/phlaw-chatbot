from chatbot.model_cache import clear_llm_cache
from chatbot.retriever import LegalRetriever
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Clear LLM caches'

    def add_arguments(self, parser):
        parser.add_argument(
            '--type',
            type=str,
            choices=['all', 'retrieval', 'model'],
            default='all',
            help='Type of cache to clear (default: all)'
        )

    def handle(self, *args, **options):
        cache_type = options['type']
        
        self.stdout.write(self.style.SUCCESS('🧹 Starting cache clearing...'))
        
        
        if cache_type in ['all', 'retrieval']:
            # Clear retrieval cache for both collections
            for collection in ['jurisprudence']:
                try:
                    retriever = LegalRetriever(collection=collection)
                    retriever.clear_cache()
                    self.stdout.write(self.style.SUCCESS(f'✅ Retrieval cache cleared for {collection}'))
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'⚠️ Could not clear retrieval cache for {collection}: {e}'))
        
        if cache_type in ['all', 'model']:
            clear_llm_cache()
            self.stdout.write(self.style.SUCCESS('✅ LLM model cache cleared'))
        
        self.stdout.write(self.style.SUCCESS('🎉 Cache clearing completed!'))
