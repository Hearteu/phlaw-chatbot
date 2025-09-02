# PHLaw Chatbot Performance Optimization Guide

## 🚀 Overview

This guide explains the performance optimizations implemented to significantly improve the speed of the PHLaw Chatbot. The original system was taking **16+ seconds** for responses, which has been optimized to target **under 5 seconds**.

## 📊 Performance Issues Identified

### Before Optimization:
- **Model Loading**: 4.65 seconds every time
- **Token Generation**: 84.42 ms per token (extremely slow)
- **Total Response Time**: 16.1 seconds for 1203 tokens
- **GPU Memory Usage**: 4.47 GB inefficiently used

### Root Causes:
1. **Model reloading**: LLM was loaded fresh on every request
2. **Suboptimal configuration**: Default settings not tuned for speed
3. **Large context window**: 4096 tokens causing memory overhead
4. **Inefficient generation parameters**: Too many tokens and slow sampling

## 🔧 Optimizations Implemented

### 1. LLM Configuration Optimization (`generator.py`)

#### Context Window Reduction
```python
"n_ctx": 2048,  # Reduced from 4096 for faster processing
```

#### GPU Optimization
```python
"n_gpu_layers": -1,        # Use all available GPU layers
"n_batch": 1024,           # Larger batch size for GPU efficiency
"n_ubatch": 512,           # Unified batch size
"mul_mat_q": True,         # Enable matrix multiplication optimization
"f16_kv": True,            # Use FP16 for key-value cache
"flash_attn": True,        # Enable flash attention if available
```

#### Memory Management
```python
"use_mmap": True,          # Use memory mapping
"use_mlock": False,        # Disable memory locking for speed
"low_vram": False,         # Disable low VRAM mode for better performance
```

#### Threading Optimization
```python
"n_threads": 12,           # Increased from 8 for better CPU utilization
```

### 2. Generation Parameters Optimization

#### Token Reduction
```python
max_tokens=512,             # Reduced from 900 for faster generation
```

#### Sampling Optimization
```python
temperature=0.3,            # Reduced from 0.4 for more focused responses
top_p=0.85,                # Reduced from 0.9 for faster sampling
top_k=40,                  # Limit top-k for faster sampling
tfs_z=0.95,                # Tail free sampling for speed
typical_p=0.95,            # Typical sampling for speed
```

#### Performance Flags
```python
stream=False,               # Disable streaming for faster completion
echo=False,                 # Don't echo the prompt
mirostat_mode=0,           # Disable mirostat for speed
```

### 3. Model Caching

#### Lazy Loading with Caching
```python
_model_loaded = False

def _ensure_llm() -> Llama:
    global _llm, _model_loaded
    if _llm is None and not _model_loaded:
        # Load model only once
        _llm = Llama(**LLM_CONFIG)
        _model_loaded = True
    return _llm
```

### 4. Environment Configuration (`.env.performance`)

#### LLM Performance Settings
```bash
LLM_CONTEXT_SIZE=2048              # Reduced context for speed
LLM_THREADS=12                     # More threads for CPU utilization
LLM_BATCH_SIZE=1024                # Larger batches for GPU efficiency
LLM_FLASH_ATTENTION=true           # Enable flash attention
LLM_F16_KV=true                    # Use FP16 for key-value cache
```

#### Embedding Optimization
```bash
CHUNK_CHARS=800                    # Reduced from 1000 for speed
OVERLAP_CHARS=150                  # Reduced from 200 for speed
EMBED_BATCH_SIZE=64                # Increased from 32 for GPU efficiency
UPSERT_BATCH=512                   # Increased from 256 for speed
```

#### System Optimization
```bash
DEBUG=false                         # Disable debug for production performance
LOG_LEVEL=WARNING                  # Reduced logging for speed
ENABLE_PROGRESS_TRACKING=false     # Disable for speed
HTTP_TIMEOUT=15                    # Reduced from 30 for speed
```

## 🧪 Testing Performance

### Run Performance Test
```bash
cd backend
python test_performance.py
```

### Expected Results
- **Model Loading**: < 3 seconds (first time only)
- **Generation Time**: < 5 seconds per response
- **Overall Speed**: > 100 characters/second

### Performance Monitoring
The test script provides:
- Model loading time
- Generation speed per query
- Overall system performance
- Performance recommendations

## 📈 Expected Performance Improvements

### Speed Improvements:
- **Model Loading**: 4.65s → <3s (35% faster)
- **Token Generation**: 84.42ms → <20ms (76% faster)
- **Total Response Time**: 16.1s → <5s (69% faster)

### Memory Efficiency:
- **Context Window**: 4096 → 2048 tokens (50% reduction)
- **GPU Memory**: Better utilization with optimized batching
- **CPU Usage**: Better threading and batch processing

## 🚨 Troubleshooting

### If Performance is Still Slow:

1. **Check GPU Utilization**
   ```bash
   nvidia-smi
   ```

2. **Verify Model Loading**
   - Check if model is cached (should see "✅ LLM model loaded successfully")
   - Restart the application to clear cache

3. **Check Environment Variables**
   ```bash
   cp .env.performance .env
   ```

4. **Monitor System Resources**
   - CPU usage should be < 80%
   - GPU memory should be < 90%
   - Available RAM should be > 2GB

### Common Issues:

#### CUDA Out of Memory
- Reduce `n_batch` in LLM_CONFIG
- Enable `low_vram: True`
- Reduce `n_ctx` further

#### Slow Generation
- Check if `flash_attn` is enabled
- Verify GPU layers are being used (`n_gpu_layers: -1`)
- Reduce `max_tokens` in generation

#### High CPU Usage
- Increase `n_threads` if CPU cores available
- Check for background processes
- Verify `use_mmap: True` is set

## 🔄 Updating Configuration

### To Apply Performance Settings:
```bash
# Copy performance-optimized environment
cp .env.performance .env

# Restart the application
python manage.py runserver
```

### To Customize Further:
1. Edit `backend/chatbot/generator.py` LLM_CONFIG
2. Modify `.env.performance` settings
3. Test with `python test_performance.py`

## 📚 Additional Resources

- [Llama.cpp Performance Tuning](https://github.com/ggerganov/llama.cpp#performance-tuning)
- [CUDA Optimization Guide](https://docs.nvidia.com/cuda/)
- [PyTorch Performance Tips](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## 🤝 Contributing

To improve performance further:
1. Profile the application with `cProfile`
2. Monitor GPU/CPU usage during inference
3. Test different model quantization levels
4. Experiment with batch sizes and context windows

---

**Note**: These optimizations prioritize speed over quality. For production use, balance between speed and response quality based on your requirements.
