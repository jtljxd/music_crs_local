"""
Test script to verify the add_all_feature_v2 pipeline can be initialized.
"""

import sys
import torch

print("Testing add_all_feature_v2 pipeline initialization...")

try:
    # Test imports
    print("\n1. Testing imports...")
    from mcrs import load_crs_baseline_v2
    from mcrs.retrieval_modules import MultiChannelRetrieval
    from mcrs.reranking_modules import ThreeTowerRerankerWrapper
    print("   ✓ All imports successful")
    
    # Test configuration loading
    print("\n2. Testing configuration loading...")
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/llama1b_multi_channel_devset.yaml")
    print(f"   ✓ Config loaded: pipeline_version={config.get('pipeline_version')}")
    
    # Test factory function
    print("\n3. Testing factory function signature...")
    import inspect
    sig = inspect.signature(load_crs_baseline_v2)
    print(f"   ✓ Function signature: {len(sig.parameters)} parameters")
    
    print("\n✅ All basic tests passed!")
    print("\nNote: Full initialization requires:")
    print("  - Qwen3-Embedding-0.6B model at: /home/lijiatong06/music-crs-baselines/Qwen3-Embedding-0.6B")
    print("  - Llama-3.2-1B-Instruct model at: /home/lijiatong06/music-crs-baselines/Llama-3.2-1B-Instruct")
    print("  - Access to HuggingFace datasets (talkpl-ai/TalkPlayData-Challenge-*)")
    print("\nTo run full inference:")
    print("  python run_inference_devset_v2.py --tid llama1b_multi_channel_devset --batch_size 8")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
