# Contextual RAG Implementation Guide

This document describes the Contextual RAG (Retrieval-Augmented Generation) implementation for the Philippine Law Chatbot project.

## Overview

The Contextual RAG system implements the approach described in [Anthropic's Contextual RAG paper](https://docs.together.ai/docs/how-to-implement-contextual-rag-from-anthropic), which enhances traditional RAG by:

1. **Context Generation**: Adding explanatory context to each chunk using an LLM
2. **Hybrid Search**: Combining dense (semantic) and sparse (keyword) search
3. **Rank Fusion**: Using Reciprocal Rank Fusion (RRF) to combine results
4. **Reranking**: Improving retrieval quality with a reranker

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │───▶│   Chunking       │───▶│ Context         │
│   Ingestion     │    │   & Processing   │    │ Generation      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query Time    │◀───│   Reranking      │◀───│ Hybrid Search   │
│   Generation    │    │   & Fusion       │    │ (Vector + BM25) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. ContextualRAG Class (`contextual_rag.py`)

Main orchestrator that handles:
- Context generation for chunks using LLM
- Building hybrid indexes (vector + BM25)
- Performing hybrid search with RRF
- Reranking results

### 2. Enhanced LegalRetriever (`retriever.py`)

Extended with Contextual RAG support:
- Automatic fallback between Contextual RAG and standard retrieval
- Seamless integration with existing chat engine
- Caching for performance

### 3. Management Command (`build_contextual_rag.py`)

Command to build Contextual RAG indexes:
```bash
python manage.py build_contextual_rag --max-cases 100 --chunk-size 640
```

## Implementation Details

### Context Generation

For each chunk, the system:
1. Takes the entire document and the specific chunk
2. Uses a prompt template to generate contextual explanation
3. Prepends the context to the original chunk content

```python
CONTEXTUAL_RAG_PROMPT = """Given the document below, we want to explain what the chunk captures in the document.

{WHOLE_DOCUMENT}

Here is the chunk we want to explain:

{CHUNK_CONTENT}

Answer ONLY with a succinct explanation of the meaning of the chunk in the context of the whole document above."""
```

### Hybrid Search

1. **Vector Search**: Uses sentence transformers for semantic similarity
2. **BM25 Search**: Uses keyword-based search with rank-bm25
3. **Rank Fusion**: Combines results using Reciprocal Rank Fusion algorithm

### Reranking

Currently uses similarity-based reranking with the embedding model. In production, this could be replaced with a dedicated cross-encoder model.

## Usage

### 1. Building Indexes

First, build the Contextual RAG indexes:

```bash
# Build indexes for all cases
python manage.py build_contextual_rag

# Build indexes for limited cases (for testing)
python manage.py build_contextual_rag --max-cases 100

# Custom parameters
python manage.py build_contextual_rag --chunk-size 800 --overlap-ratio 0.2
```

### 2. Using in Code

```python
from chatbot.retriever import LegalRetriever

# Initialize with Contextual RAG
retriever = LegalRetriever(use_contextual_rag=True)

# Query
results = retriever.retrieve("contract breach", k=5)
```

### 3. Automatic Integration

The chat engine automatically uses Contextual RAG when available:

```python
from chatbot.chat_engine import chat_with_law_bot

response = chat_with_law_bot("What is due process?")
```

## Configuration

### Environment Variables

- `QDRANT_HOST`: Qdrant server host (default: localhost)
- `QDRANT_PORT`: Qdrant server port (default: 6333)

### Parameters

- `chunk_size`: Size of chunks in tokens (default: 640)
- `overlap_ratio`: Overlap between chunks (default: 0.15)
- `context_model`: LLM model for context generation
- `vector_k`: Number of vector search results (default: 150)
- `bm25_k`: Number of BM25 search results (default: 150)
- `final_k`: Final number of results after reranking (default: 20)

## Performance Considerations

### Memory Usage

- Context generation requires significant memory for large documents
- Embeddings are cached to avoid recomputation
- Consider processing documents in batches for large datasets

### Speed

- Context generation is the slowest step (requires LLM calls)
- Vector search is fast with cached embeddings
- BM25 search is very fast
- Reranking is moderately fast

### Optimization Tips

1. **Batch Processing**: Process multiple cases together
2. **Caching**: Results are cached to avoid recomputation
3. **Model Selection**: Use smaller, faster models for context generation
4. **Index Management**: Regularly update indexes as data changes

## Testing

Run the test suite to verify implementation:

```bash
cd backend
python test_contextual_rag.py
```

This will test:
- System initialization
- Retriever functionality
- Chat engine integration
- Fallback mechanisms

## Monitoring

### Index Statistics

```python
from chatbot.contextual_rag import create_contextual_rag_system

rag = create_contextual_rag_system()
stats = rag.get_index_stats()
print(stats)
```

### Performance Metrics

Monitor:
- Index building time
- Query response time
- Memory usage
- Cache hit rates

## Troubleshooting

### Common Issues

1. **Contextual RAG not available**
   - Check if indexes are built
   - Verify Qdrant connection
   - Check model availability

2. **Slow performance**
   - Reduce chunk size
   - Limit number of cases
   - Use smaller context model

3. **Memory issues**
   - Process fewer cases at once
   - Increase system memory
   - Use memory-efficient models

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Advanced Reranking**: Implement cross-encoder models
2. **Adaptive Chunking**: Dynamic chunk sizes based on content
3. **Multi-Modal**: Support for images and tables
4. **Real-time Updates**: Incremental index updates
5. **Performance Optimization**: GPU acceleration for embeddings

## References

- [Anthropic Contextual RAG Paper](https://docs.together.ai/docs/how-to-implement-contextual-rag-from-anthropic)
- [Together AI Implementation](https://docs.together.ai/docs/how-to-implement-contextual-rag-from-anthropic)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
