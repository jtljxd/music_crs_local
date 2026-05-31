#!/bin/bash
# Quick start script for add_all_feature_v2 pipeline

echo "================================================"
echo "  Add All Feature V2 - Quick Start"
echo "================================================"
echo ""

# Check if we're on the server
if [ ! -d "/home/lijiatong06/music-crs-baselines" ]; then
    echo "⚠️  Warning: Expected model directory not found"
    echo "   This script is designed for the server environment"
    echo "   Expected: /home/lijiatong06/music-crs-baselines/"
    echo ""
fi

# Step 1: Verify models
echo "Step 1: Verifying model paths..."
QWEN_PATH="/home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B"
LLAMA_PATH="/home/lijiatong06/music-crs-baselines/Llama-3.2-1B-Instruct"

if [ -d "$QWEN_PATH" ]; then
    echo "  ✓ Qwen3-Embedding-0.6B found"
else
    echo "  ✗ Qwen3-Embedding-0.6B not found at $QWEN_PATH"
fi

if [ -d "$LLAMA_PATH" ]; then
    echo "  ✓ Llama-3.2-1B-Instruct found"
else
    echo "  ✗ Llama-3.2-1B-Instruct not found at $LLAMA_PATH"
fi
echo ""

# Step 2: Check Python environment
echo "Step 2: Checking Python environment..."
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch not found"
python3 -c "import transformers; print(f'  ✓ Transformers {transformers.__version__}')" 2>/dev/null || echo "  ✗ Transformers not found"
python3 -c "import datasets; print(f'  ✓ Datasets {datasets.__version__}')" 2>/dev/null || echo "  ✗ Datasets not found"
python3 -c "import omegaconf; print(f'  ✓ OmegaConf {omegaconf.__version__}')" 2>/dev/null || echo "  ✗ OmegaConf not found"
echo ""

# Step 3: Validate code
echo "Step 3: Validating Python code..."
python3 -m py_compile mcrs/retrieval_modules/multi_channel_retrieval.py && echo "  ✓ multi_channel_retrieval.py" || echo "  ✗ multi_channel_retrieval.py"
python3 -m py_compile mcrs/reranking_modules/three_tower_reranker.py && echo "  ✓ three_tower_reranker.py" || echo "  ✗ three_tower_reranker.py"
python3 -m py_compile mcrs/crs_baseline_v2.py && echo "  ✓ crs_baseline_v2.py" || echo "  ✗ crs_baseline_v2.py"
python3 -m py_compile run_inference_devset_v2.py && echo "  ✓ run_inference_devset_v2.py" || echo "  ✗ run_inference_devset_v2.py"
echo ""

# Step 4: Show configuration
echo "Step 4: Configuration summary..."
echo "  Config file: config/llama1b_multi_channel_devset.yaml"
echo "  Pipeline: V2 (multi-channel + three-tower)"
echo "  Retrieval: ~350 candidates from 4 channels"
echo "  Reranking: Top 20 via three-tower model"
echo "  LLM: Llama-3.2-1B-Instruct"
echo ""

# Step 5: Run options
echo "Step 5: Ready to run!"
echo ""
echo "Option A - Run inference (recommended batch_size=8):"
echo "  python run_inference_devset_v2.py --tid llama1b_multi_channel_devset --batch_size 8"
echo ""
echo "Option B - Run with smaller batch (if GPU memory limited):"
echo "  python run_inference_devset_v2.py --tid llama1b_multi_channel_devset --batch_size 4"
echo ""
echo "Option C - Test imports only:"
echo "  python test_v2_pipeline.py"
echo ""
echo "Note: First run will build cache (~5-10 minutes)"
echo "      Subsequent runs will be faster using cached data"
echo ""
echo "================================================"
