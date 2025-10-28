# Legal RoBERTa Fine-tuning Guide

## 🎯 Overview

This guide will help you fine-tune the Saibo Legal RoBERTa model on your 21k Philippine jurisprudence cases to achieve 75-85% classification accuracy.

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: RTX 4050 6GB VRAM ✅
- **RAM**: 16GB+ recommended
- **Storage**: 5GB free space for model and data

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch with CUDA support

## 🚀 Step-by-Step Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements_finetuning.txt
```

### 2. Verify GPU Setup

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4050
```

### 3. Prepare Your Data

Make sure you have jurisprudence data:
```bash
# Check if data exists
ls -la data/cases.jsonl.gz
```

If no data exists, run the crawler first:
```bash
python crawler.py
```

### 4. Run Fine-tuning

```bash
python fine_tune_legal_roberta.py
```

## ⏱️ Expected Timeline

### Training Time: 2-3 hours
- **Epoch 1**: ~45 minutes
- **Epoch 2**: ~45 minutes  
- **Epoch 3**: ~45 minutes
- **Evaluation**: ~15 minutes

### Memory Usage
- **VRAM**: ~5-6GB (fits RTX 4050)
- **RAM**: ~8-12GB
- **Storage**: ~3GB for model

## 📊 Expected Results

### Before Fine-tuning:
- Classification accuracy: ~55%
- Confidence scores: 0.546-0.552
- Poor administrative case handling

### After Fine-tuning:
- Classification accuracy: 75-85% ✅
- Higher confidence scores: 0.75-0.85
- Proper legal category classification
- Better Philippine jurisprudence understanding

## 🔧 Configuration Details

### Model Settings (from official documentation):
```python
LEARNING_RATE = 5e-5  # Official recommendation
EPOCHS = 3           # Official recommendation
BATCH_SIZE = 8       # Optimized for RTX 4050
GRADIENT_ACCUMULATION = 8  # Effective batch size = 64
```

### Legal Categories:
- **Case Types**: 24 categories (civil, criminal, administrative, etc.)
- **Legal Areas**: 24 categories (criminal_law, civil_law, etc.)
- **Document Sections**: 18 categories (facts, issues, ruling, etc.)
- **Complexity Levels**: 4 categories
- **Jurisdiction Levels**: 8 categories
- **Total Labels**: 78 categories

## 🚨 Troubleshooting

### Common Issues:

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in CONFIG
BATCH_SIZE = 4  # Instead of 8
GRADIENT_ACCUMULATION = 16  # Increase to maintain effective batch size
```

#### 2. Slow Training
```bash
# Enable mixed precision (already enabled)
fp16 = True
```

#### 3. Model Not Loading
```bash
# Check model path
ls -la legal_roberta_finetuned/
```

## 📈 Monitoring Progress

### Training Logs:
- Loss decreases over epochs
- F1 score increases
- Validation metrics improve

### Expected Metrics:
- **Training Loss**: 0.8 → 0.3
- **Validation F1**: 0.55 → 0.80
- **Accuracy**: 0.55 → 0.85

## 🔄 Integration Steps

### 1. After Fine-tuning Completes:

```bash
python integrate_finetuned_model.py
```

### 2. Update Your Embedding Pipeline:

The fine-tuned model will be automatically integrated into your existing system.

### 3. Re-run Embedding:

```bash
python embed.py
```

## 📁 File Structure After Fine-tuning:

```
backend/
├── legal_roberta_finetuned/          # Fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── eval_results.json
├── fine_tune_legal_roberta.py        # Training script
├── integrate_finetuned_model.py      # Integration script
└── requirements_finetuning.txt       # Dependencies
```

## 🎯 Success Criteria

Fine-tuning is successful when:
- ✅ Training completes without errors
- ✅ Validation F1 score > 0.75
- ✅ Classification accuracy > 0.80
- ✅ Administrative cases properly classified
- ✅ Legal categories accurately detected

## 🆘 Support

If you encounter issues:
1. Check GPU memory usage: `nvidia-smi`
2. Verify data format: Check JSONL structure
3. Monitor training logs for errors
4. Ensure all dependencies are installed

## 🚀 Next Steps After Fine-tuning

1. **Test the model** with sample cases
2. **Re-run embedding** with improved classification
3. **Evaluate chatbot performance** 
4. **Monitor classification accuracy** in production
5. **Fine-tune further** if needed for specific case types

---

**Expected Outcome**: Your Legal RoBERTa model will be specifically trained on Philippine jurisprudence, achieving 75-85% classification accuracy and significantly improving your chatbot's legal understanding capabilities.
