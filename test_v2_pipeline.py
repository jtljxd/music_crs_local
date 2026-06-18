"""
Test script to verify the CRS baseline pipeline can be initialized.
"""

import sys
import torch

print("Testing CRS baseline pipeline initialization...")

try:
    # Test imports
    print("\n1. Testing imports...")
    from mcrs import load_crs_baseline, load_crs_baseline_v2
    from mcrs.retrieval_modules import load_retrieval_module
    print("   ✓ All imports successful")

    # Test configuration loading
    print("\n2. Testing configuration loading...")
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/llama1b_bm25_devset.yaml")
    print(f"   ✓ Config loaded: retrieval_type={config.get('retrieval_type')}")

    # Test factory function
    print("\n3. Testing factory function signatures...")
    import inspect
    sig_v1 = inspect.signature(load_crs_baseline)
    sig_v2 = inspect.signature(load_crs_baseline_v2)
    print(f"   ✓ load_crs_baseline: {len(sig_v1.parameters)} parameters")
    print(f"   ✓ load_crs_baseline_v2: {len(sig_v2.parameters)} parameters")

    print("\n✅ All basic tests passed!")
    print("\nNote: Full initialization requires:")
    print("  - Llama-3.2-1B-Instruct model")
    print("  - Access to HuggingFace datasets (talkpl-ai/TalkPlayData-Challenge-*)")
    print("\nTo run full inference:")
    print("  python run_inference_devset.py --retrieval_type bm25")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
