# build_contextual_rag.py — Management command to build Contextual RAG indexes
import json
import os

from chatbot.contextual_rag import create_contextual_rag_system
from chatbot.retriever import load_case_from_jsonl
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Build Contextual RAG indexes with hybrid search and reranking'

    def add_arguments(self, parser):
        parser.add_argument(
            '--collection',
            type=str,
            default='jurisprudence',
            help='Qdrant collection name (default: jurisprudence)'
        )
        parser.add_argument(
            '--chunk-size',
            type=int,
            default=640,
            help='Chunk size in tokens (default: 640)'
        )
        parser.add_argument(
            '--overlap-ratio',
            type=float,
            default=0.15,
            help='Overlap ratio between chunks (default: 0.15)'
        )
        parser.add_argument(
            '--max-cases',
            type=int,
            default=None,
            help='Maximum number of cases to process (default: all)'
        )
        parser.add_argument(
            '--data-file',
            type=str,
            default=None,
            help='Path to JSONL data file (default: auto-detect)'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting Contextual RAG index building...')
        )
        
        # Get parameters
        collection = options['collection']
        chunk_size = options['chunk_size']
        overlap_ratio = options['overlap_ratio']
        max_cases = options['max_cases']
        data_file = options['data_file']
        
        # Initialize Contextual RAG system
        try:
            contextual_rag = create_contextual_rag_system(collection=collection)
            contextual_rag.chunk_size = chunk_size
            contextual_rag.overlap_ratio = overlap_ratio
            
            self.stdout.write(
                self.style.SUCCESS(f'✅ Contextual RAG system initialized for collection: {collection}')
            )
        except Exception as e:
            raise CommandError(f'Failed to initialize Contextual RAG system: {e}')
        
        # Load cases from JSONL
        try:
            if data_file is None:
                # Auto-detect data file
                data_file = os.path.join(
                    os.path.dirname(__file__), 
                    "..", "..", "..", "data", "cases.jsonl.gz"
                )
            
            if not os.path.exists(data_file):
                raise CommandError(f'Data file not found: {data_file}')
            
            cases = self._load_cases_from_jsonl(data_file, max_cases)
            self.stdout.write(
                self.style.SUCCESS(f'✅ Loaded {len(cases)} cases from {data_file}')
            )
            
        except Exception as e:
            raise CommandError(f'Failed to load cases: {e}')
        
        # Build indexes
        try:
            self.stdout.write(
                self.style.WARNING('🔄 Building hybrid indexes (this may take a while)...')
            )
            
            contextual_rag.build_hybrid_indexes(cases)
            
            # Get stats
            stats = contextual_rag.get_index_stats()
            
            self.stdout.write(
                self.style.SUCCESS('✅ Contextual RAG indexes built successfully!')
            )
            
            # Display stats
            self.stdout.write('\n📊 Index Statistics:')
            for key, value in stats.items():
                self.stdout.write(f'  {key}: {value}')
                
        except Exception as e:
            raise CommandError(f'Failed to build indexes: {e}')
    
    def _load_cases_from_jsonl(self, data_file: str, max_cases: int = None) -> list:
        """Load cases from JSONL file"""
        import gzip
        
        cases = []
        count = 0
        
        try:
            with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        case = json.loads(line)
                        cases.append(case)
                        count += 1
                        
                        if count % 100 == 0:
                            self.stdout.write(f'  Loaded {count} cases...')
                        
                        if max_cases and count >= max_cases:
                            break
                            
                    except json.JSONDecodeError as e:
                        self.stdout.write(
                            self.style.WARNING(f'⚠️ Skipping invalid JSON at line {line_num}: {e}')
                        )
                        continue
                        
        except Exception as e:
            raise CommandError(f'Error reading JSONL file: {e}')
        
        return cases
